import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CustomDataset
from functools import partial
# from models import vgg19
from Networks import ALTGVT
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import wandb



def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


def train_collate_custom_dataset(batch, args):
    transposed_batch = list(zip(*batch))
    patches_per_image = 4 if args.finetuning else 1

    num_of_images = len(transposed_batch[0])
    batch_dimension = patches_per_image * num_of_images # TODO harcoed, fix later 4 because each image generate 4 patches 256x256

    images = torch.stack(transposed_batch[0], 0) # shape batch_dim, num_patches_per_image, channel, w, h
    images = torch.reshape(images, (batch_dimension, 3, 256,256)) # TODO fix dimension
    points_batch = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    points = []
    for b in points_batch:
        for p in b:
            points.append(p)

    gt_discretes = torch.stack(transposed_batch[2], 0)
    gt_discretes = torch.reshape(gt_discretes, (batch_dimension, 1, 256,256)) # TODO fix dimension
    
    foregrounds = torch.stack(transposed_batch[3], 0)
    foregrounds = torch.reshape(foregrounds, (batch_dimension, 256,256)) # TODO fix dimension
    return images, points, gt_discretes, foregrounds


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = (
            "ALTGVT/{}_12-1-input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}".format(
                args.run_name,
                args.crop_size,
                args.wot,
                args.wtv,
                args.reg,
                args.num_of_iter_in_ot,
                args.norm_cood,
            )
        )

        self.save_dir = os.path.join("ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str))
        )
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info("using {} gpus".format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        if args.dataset.lower() == "qnrf":
            self.datasets = {
                x: Crowd_qnrf(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "nwpu":
            self.datasets = {
                x: Crowd_nwpu(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "sha" or args.dataset.lower() == "shb":
            self.datasets = {
                "train": Crowd_sh(
                    os.path.join(args.data_dir, "train_data"),
                    args.crop_size,
                    downsample_ratio,
                    "train",
                ),
                "val": Crowd_sh(
                    os.path.join(args.data_dir, "test_data"),
                    args.crop_size,
                    downsample_ratio,
                    "val",
                ),
            }
        elif args.dataset.lower() == "custom":
            self.datasets = {
                "train": CustomDataset(
                    args.data_dir, args.crop_size, downsample_ratio, method="train", finetuning=self.args.finetuning
                ),
                "val": CustomDataset(
                    args.data_dir, args.crop_size, downsample_ratio, method="valid", finetuning=self.args.finetuning
                ),
            }
            print("[FINETUNIG]:", bool(self.args.finetuning))
        else:
            raise NotImplementedError

        if args.dataset.lower() == "custom": 
            self.dataloaders = {
                x: DataLoader(
                    self.datasets[x],
                    collate_fn=partial(train_collate_custom_dataset, args=args),
                    batch_size=(args.batch_size),
                    shuffle=True,
                    num_workers=args.num_workers * self.device_count,
                    pin_memory=(True if x == "train" else False),
                )
                for x in ["train", "val"]
            }
        else:
            self.dataloaders = {
                x: DataLoader(
                    self.datasets[x],
                    collate_fn=(train_collate if x ==
                                "train" else default_collate),
                    batch_size=(args.batch_size if x == "train" else 1),
                    shuffle=(True if x == "train" else False),
                    num_workers=args.num_workers * self.device_count,
                    pin_memory=(True if x == "train" else False),
                )
                for x in ["train", "val"]
            }
            
        self.model = ALTGVT.alt_gvt_large(pretrained=True)
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        print("[PARAMS] #:",  sum(map(torch.numel, self.model.parameters())))
        self.start_epoch = 0

        # check if wandb has to log
        if args.wandb:
            self.wandb_run = wandb.init(
            config=args, project="CTTrans", name=args.run_name
        )
        else : 
            wandb.init(mode="disabled")
    

        if args.resume:
            self.logger.info("loading pretrained model from " + args.resume)
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(
                    torch.load(args.resume, self.device))
        else:
            self.logger.info("random initialization")

        self.ot_loss = OT_Loss(
            args.crop_size,
            1,# downsample_ratio
            args.norm_cood,
            self.device,
            args.num_of_iter_in_ot,
            args.reg,
        )
        self.tv_loss = nn.L1Loss(reduction="none").to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf

        self.foreground_loss = nn.BCELoss().to(self.device)

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info(
                "-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5
            )
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_segmentation_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, points, gt_discrete, foreground_gt) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            foreground_gt = foreground_gt.unsqueeze(1).to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed, output_foreground = self.model(inputs)

                # compute foreground loss
                foreground_out_loss = self.foreground_loss(output_foreground, foreground_gt)
                foreground_out_loss = foreground_out_loss * self.args.lambda_segmentation

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(
                    outputs_normed, outputs, points
                )
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(
                    outputs.sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().to(self.device),
                )
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = (
                    torch.from_numpy(gd_count)
                    .float()
                    .to(self.device)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (
                    self.tv_loss(outputs_normed, gt_discrete_normed)
                    .sum(1)
                    .sum(1)
                    .sum(1)
                    * torch.from_numpy(gd_count).float().to(self.device)
                ).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)
                epoch_segmentation_loss.update(foreground_out_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss + foreground_out_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (
                    torch.sum(outputs.view(N, -1),
                              dim=1).detach().cpu().numpy()
                )
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

        # log wandb
        wandb.log(
            {
                "train/TOTAL_loss": epoch_loss.get_avg(),
                "train/count_loss":epoch_count_loss.get_avg(),
                "train/tv_loss": epoch_tv_loss.get_avg(),
                "train/segmentation_loss" : epoch_segmentation_loss.get_avg()
            },
            step=self.epoch,
        )

        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, "
            "Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch,
                epoch_loss.get_avg(),
                epoch_ot_loss.get_avg(),
                epoch_wd.get_avg(),
                epoch_ot_obj_value.get_avg(),
                epoch_count_loss.get_avg(),
                epoch_tv_loss.get_avg(),
                np.sqrt(epoch_mse.get_avg()),
                epoch_mae.get_avg(),
                time.time() - epoch_start,
            )
        )
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(
            self.save_dir, "{}_ckpt.tar".format(self.epoch))
        torch.save(
            {
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": model_state_dic,
            },
            save_path,
        )
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_total_loss = AverageMeter()
        epoch_segmentation_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode

        for step, (inputs, points, gt_discrete, foreground_gt) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            foreground_gt = foreground_gt.unsqueeze(1).to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)
            
            with torch.no_grad():
                outputs, outputs_normed, output_foreground = self.model(inputs)
                # compute foreground loss
                foreground_out_loss = self.foreground_loss(output_foreground, foreground_gt)
                foreground_out_loss = foreground_out_loss * self.args.lambda_segmentation

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(
                    outputs_normed, outputs, points
                )
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot

                # Compute counting loss.
                count_loss = self.mae(
                    outputs.sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().to(self.device),
                )

                # Compute TV loss.
                gd_count_tensor = (
                    torch.from_numpy(gd_count)
                    .float()
                    .to(self.device)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (
                    self.tv_loss(outputs_normed, gt_discrete_normed)
                    .sum(1)
                    .sum(1)
                    .sum(1)
                    * torch.from_numpy(gd_count).float().to(self.device)
                ).mean(0) * self.args.wtv

                loss = ot_loss + count_loss + tv_loss + foreground_out_loss
                
                pred_count = (
                    torch.sum(outputs.view(N, -1),
                              dim=1).detach().cpu().numpy()
                )
                pred_err = pred_count - gd_count
                
                epoch_total_loss.update(loss.item(), N)
                epoch_segmentation_loss.update(foreground_out_loss.item(), N)
                epoch_count_loss.update(count_loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

        mae = epoch_mae.get_avg()
        mse = np.sqrt(epoch_mse.get_avg())
        # log to wandb
        wandb.log(
            {
                "val/TOTAL_loss": epoch_total_loss.get_avg(),
                "val/count_loss": epoch_count_loss.get_avg(),
                "val/segmentation_loss" : epoch_segmentation_loss.get_avg(),
                "val/MAE" : mae,
                "val/MSE" : mse,
            },step=self.epoch,
        )

        model_state_dic = self.model.state_dict()
        
        print("Comparison Validation mae", mae,  self.best_mae)
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                "save best mse {:.2f} mae {:.2f} model epoch {}".format(
                    self.best_mse, self.best_mae, self.epoch
                )
            )
            print("Saving best model at {} epoch".format(self.epoch))
            model_path = os.path.join(
                self.save_dir, "best_model_mae-{:.2f}_epoch-{}.pth".format(
                    self.best_mae, self.epoch)
            )
            torch.save(
                model_state_dic,
                model_path,
            )

            if args.wandb:
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_path)
                
                self.wandb_run.log_artifact(artifact)


def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h // factor * factor), int(w // factor * factor)
    img_tensor = F.interpolate(
        img_tensor, (h, w), mode="bilinear", align_corners=True)

    return img_tensor


def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(
                img_tensor,
                (min_size, int(min_size / h * w)),
                mode="bilinear",
                align_corners=True,
            )
        else:
            img_tensor = F.interpolate(
                img_tensor,
                (int(min_size / w * h), min_size),
                mode="bilinear",
                align_corners=True,
            )
    return img_tensor


if __name__ == "__main__":
    import torch

    print(torch.__file__)
    x = torch.ones(1, 3, 768, 1152)
    y = tensor_spilt(x)
    print(y.size())
