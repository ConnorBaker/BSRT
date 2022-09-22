import os
from pathlib import Path
import model.bsrt as bsrt
from option import Config
from pytorch_lightning.trainer import Trainer
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies.bagua import BaguaStrategy
from data_modules.zurich_raw2rgb_data_module import ZurichRaw2RgbDataModule
from data_modules.synthetic_burst_data_module import SyntheticBurstDataModule


def main(config: Config):
    pl.seed_everything(config.seed, workers=True)

    num_workers = os.cpu_count()
    num_workers = num_workers // 2 if num_workers is not None else 0

    _model = bsrt.make_model(config)

    train_module = ZurichRaw2RgbDataModule(
        data_dir=Path(config.data_dir),
        batch_size=config.batch_size,
        num_workers=num_workers,
    )
    train_module.prepare_data()
    train_module.setup()
    train_data = train_module.train_data

    data_module = SyntheticBurstDataModule(
        dataset=train_data,
        data_dir=Path(config.data_dir),
        burst_size=config.burst_size,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )

    # auto_lr_find=True?
    trainer = Trainer(
        deterministic=False,
        benchmark=True,
        accelerator="auto",
        strategy=BaguaStrategy(),
        profiler="simple",
    )
    trainer.fit(_model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSRT")
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--data_type",
        type=str,
        choices=("synthetic", "real"),
        help="whether operating on synthetic or real data",
    )
    parser.add_argument(
        "--n_resblocks", type=int, default=16, help="number of residual blocks"
    )
    parser.add_argument(
        "--n_feats", type=int, default=64, help="number of feature maps"
    )
    parser.add_argument(
        "--n_colors", type=int, default=3, help="number of color channels to use"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--burst_size", type=int, default=14, help="burst size, max 14")
    parser.add_argument(
        "--burst_channel", type=int, default=4, help="RAW channel, default:4"
    )
    parser.add_argument(
        "--swinfeature",
        action="store_true",
        help="use swin transformer to extract features",
    )
    parser.add_argument(
        "--model_level",
        type=str,
        default="S",
        choices=("S", "L"),
        help="S: small, L: large",
    )

    ################## fine-tune ##################
    parser.add_argument("--finetune", action="store_true", help="finetune model")
    parser.add_argument(
        "--finetune_align", action="store_true", help="finetune alignment module"
    )
    parser.add_argument(
        "--finetune_swin", action="store_true", help="finetune swin trans module"
    )
    parser.add_argument(
        "--finetune_conv", action="store_true", help="finetune rest convs"
    )
    parser.add_argument(
        "--finetune_prelayer",
        action="store_true",
        help="finetune finetune pre feature extract layer",
    )
    parser.add_argument(
        "--finetune_upconv", action="store_true", help="finetune finetune up conv layer"
    )
    parser.add_argument(
        "--finetune_spynet", action="store_true", help="finetune finetune up conv layer"
    )

    # Hardware specifications
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="use use_checkpoint in swin transformer",
    )

    # Data specifications
    parser.add_argument(
        "--data_dir",
        type=str,
        help="dataset directory",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="demo image directory"
    )
    parser.add_argument("--scale", type=int, default=4, help="super resolution scale")
    parser.add_argument("--patch_size", type=int, default=256, help="output patch size")
    parser.add_argument("--rgb_range", type=int, default=1, help="maximum value of RGB")

    # Model specifications
    parser.add_argument("--act", type=str, default="relu", help="activation function")
    parser.add_argument("--res_scale", type=float, default=1, help="residual scaling")
    parser.add_argument(
        "--shift_mean",
        type=bool,
        default=True,
        help="subtract pixel mean from the input",
    )
    parser.add_argument(
        "--dilation", action="store_true", help="use dilated convolution"
    )

    # Option for Residual channel attention network (RCAN)
    parser.add_argument(
        "--n_resgroups", type=int, default=20, help="number of residual groups"
    )
    parser.add_argument(
        "--reduction", type=int, default=16, help="number of feature maps reduction"
    )
    parser.add_argument("--DA", action="store_true", help="use Dual Attention")
    parser.add_argument("--CA", action="store_true", help="use Channel Attention")
    parser.add_argument("--non_local", action="store_true", help="use Dual Attention")

    # Training specifications
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="input batch size for training"
    )
    parser.add_argument(
        "--gan_k", type=int, default=1, help="k value for adversarial loss"
    )

    # Optimization specifications
    parser.add_argument(
        "--decay", type=str, default="40-80", help="learning rate decay type"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="learning rate decay factor for step decay",
    )
    parser.add_argument(
        "--optimizer",
        default="ADAM",
        choices=("SGD", "ADAM", "RMSprop"),
        help="optimizer to use (SGD | ADAM | RMSprop)",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="ADAM beta")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="ADAM epsilon for numerical stability",
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--gclip",
        type=float,
        default=0,
        help="gradient clipping threshold (0 = no clipping)",
    )

    # Loss specifications
    parser.add_argument(
        "--loss", type=str, default="1*L1", help="loss function configuration"
    )
    parser.add_argument(
        "--skip_threshold",
        type=float,
        default="1e8",
        help="skipping batch that has large error",
    )

    args = parser.parse_args()
    config = Config(**args.__dict__)
    main(config)
