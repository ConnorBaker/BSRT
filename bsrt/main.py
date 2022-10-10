from argparse import ArgumentParser
from importlib import resources
from datasets.synthetic_burst.train_dataset import TrainData, TrainDataProcessor
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset
from option import Config
from pandas import DataFrame
from pathlib import Path
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics.metric import Metric
from typing import Any
from utility import make_optimizer, make_model, make_loss_fn, make_psnr_fn
import logging
import optuna
import os
import torch
import torch.backends.cuda
import torch.backends.cudnn
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# configure logging at the root level of Lightning
logger = logging.getLogger("pytorch_lightning")
logger.setLevel(logging.INFO)


os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TORCH_CUDNN_V8_API_DEBUG"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class BSRTModule(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = make_model(cfg)
        self.loss_fn = make_loss_fn(cfg.loss, cfg.data_type)
        self.psnr_fn = make_psnr_fn(cfg.data_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: TrainData, batch_idx: int) -> torch.Tensor:
        bursts = batch["burst"]
        gts = batch["gt"]
        srs = self(bursts)
        loss = self.loss_fn(srs, gts)
        psnr, ssim, lpips = self.psnr_fn(srs, gts)
        self.log("train_loss", loss.item())
        self.log("val_psnr", psnr.item())
        self.log("val_ssim", ssim.item())
        self.log("val_lpips", lpips.item())
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        psnr, ssim, lpips = self.psnr_fn(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)
        self.log("val_lpips", lpips)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return make_optimizer(self.cfg, self.model)


def train_setup(cfg: Config) -> None:
    """Set up the training environment."""
    datamodule = ZurichRaw2RgbDataset(data_dir=Path(cfg.data_dir), batch_size=cfg.batch_size, transform=TrainDataProcessor(burst_size=cfg.burst_size, crop_sz=cfg.patch_size))

    module = BSRTModule(cfg)
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        precision="bf16",
    )
    trainer.fit(module, datamodule)

if __name__ == "__main__":
    parser = ArgumentParser(description="BSRT")

    ################## bsrt ##################
    bsrt_group = parser.add_argument_group("bsrt")
    bsrt_group.add_argument("--seed", type=int, default=1, help="random seed")
    bsrt_group.add_argument(
        "--data_type",
        type=str,
        choices=("synthetic", "real"),
        help="whether operating on synthetic or real data",
    )
    bsrt_group.add_argument(
        "--n_resblocks", type=int, default=16, help="number of residual blocks"
    )
    bsrt_group.add_argument(
        "--n_feats", type=int, default=64, help="number of feature maps"
    )
    bsrt_group.add_argument(
        "--n_colors", type=int, default=3, help="number of color channels to use"
    )
    bsrt_group.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    bsrt_group.add_argument(
        "--burst_size", type=int, default=14, help="burst size, max 14"
    )
    bsrt_group.add_argument(
        "--burst_channel", type=int, default=4, help="RAW channel, default:4"
    )
    bsrt_group.add_argument(
        "--swinfeature",
        action="store_true",
        help="use swin transformer to extract features",
    )
    bsrt_group.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="use use_checkpoint in swin transformer",
    )
    bsrt_group.add_argument(
        "--model_level",
        type=str,
        default="S",
        choices=("S", "L"),
        help="S: small, L: large",
    )

    ################## fine-tune ##################
    fine_tune_group = parser.add_argument_group("fine-tune")
    fine_tune_group.add_argument(
        "--finetune", action="store_true", help="finetune model"
    )
    fine_tune_group.add_argument(
        "--finetune_align", action="store_true", help="finetune alignment module"
    )
    fine_tune_group.add_argument(
        "--finetune_swin", action="store_true", help="finetune swin trans module"
    )
    fine_tune_group.add_argument(
        "--finetune_conv", action="store_true", help="finetune rest convs"
    )
    fine_tune_group.add_argument(
        "--finetune_prelayer",
        action="store_true",
        help="finetune finetune pre feature extract layer",
    )
    fine_tune_group.add_argument(
        "--finetune_upconv", action="store_true", help="finetune finetune up conv layer"
    )
    fine_tune_group.add_argument(
        "--finetune_spynet", action="store_true", help="finetune finetune up conv layer"
    )

    ################## dataset ##################
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        help="dataset directory",
    )
    dataset_group.add_argument(
        "--mode", type=str, default="train", help="demo image directory"
    )
    dataset_group.add_argument(
        "--scale", type=int, default=4, help="super resolution scale"
    )
    dataset_group.add_argument(
        "--patch_size", type=int, default=256, help="output patch size"
    )
    dataset_group.add_argument(
        "--rgb_range", type=int, default=1, help="maximum value of RGB"
    )

    ################## rcan ##################
    rcan_group = parser.add_argument_group("rcan")
    rcan_group.add_argument(
        "--non_local", action="store_true", help="use Dual Attention"
    )

    ################## training ##################
    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--batch_size", type=int, default=8, help="input batch size for training"
    )
    train_group.add_argument(
        "--epochs", type=int, default=16, help="number of epochs for training"
    )
    train_group.add_argument(
        "--gan_k", type=int, default=1, help="k value for adversarial loss"
    )

    ################## optimization ##################
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--decay_milestones",
        nargs="+",
        type=int,
        default=["40", "80"],
        help="learning rate decay type",
    )
    optimization_group.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="learning rate decay factor for step decay",
    )
    optimization_group.add_argument(
        "--optimizer",
        default="ADAM",
        choices=("SGD", "ADAM", "RMSprop"),
        help="optimizer to use (SGD | ADAM | RMSprop)",
    )
    optimization_group.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum"
    )
    optimization_group.add_argument(
        "--beta_gradient", type=float, default=0.9, help="ADAM beta for gradient"
    )

    optimization_group.add_argument(
        "--beta_square", type=float, default=0.999, help="ADAM beta for square"
    )
    optimization_group.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="ADAM epsilon for numerical stability",
    )
    optimization_group.add_argument(
        "--weight_decay", type=float, default=0, help="weight decay"
    )

    ################## loss ##################
    loss_group = parser.add_argument_group("loss")
    loss_group.add_argument(
        "--loss",
        type=str,
        default="L1",
        choices=("L1", "MSE", "CB", "MSSIM"),
        help="loss function configuration",
    )

    args = parser.parse_args()
    cfg = Config(**args.__dict__)
    train_setup(cfg)
