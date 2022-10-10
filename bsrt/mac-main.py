import os
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from datasets.synthetic_burst.test_dataset import TestDataset
from datasets.zurich_raw2rgb_dataset import ZuricRaw2RgbData
from datasets.zurich_raw2rgb_dataset import ImageFolderData
from datasets.synthetic_burst.train_dataset import TrainData, TrainDataset
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset
import model.bsrt as bsrt
from option import Config
from pytorch_lightning.trainer import Trainer
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
from ray import train
from ray.data.context import DatasetContext
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.dataset import Dataset

import numpy as np

# from data_modules.zurich_raw2rgb_data_module import ZurichRaw2RgbDataModule
# from data_modules.synthetic_burst_data_module import SyntheticBurstDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from ray_lightning import RayStrategy
from ray import tune
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import ScalingConfig, DatasetConfig
from ray.train.torch import TorchTrainer


def train(config):
    cfg: Config = config["cfg"]
    args: Namespace = config["args"]
    pl.seed_everything(cfg.seed, workers=True)

    num_workers = os.cpu_count()
    num_workers = num_workers // 2 if num_workers is not None else 0

    # from torchmetrics.utilities.checks import check_forward_full_state_property
    # import metrics.l1
    # check_forward_full_state_property(metric_class=metrics.l1.L1, init_args = {"boundary_ignore": 16}, input_args={"pred": torch.rand(1, 3, 64, 64), "gt": torch.rand(1, 3, 64, 64)})

    _model = bsrt.make_model(cfg)
    # Create the Tune Reporting Callback
    # session.report({"loss": 1.0}, checkpoint=Checkpoint.from_directory("/Users/connorbaker/Packages/BSRT/bsrt/checkpoints"))
    # metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    # trc = TuneReportCallback(metrics, on="validation_end")

    zrr_dataset = ZurichRaw2RgbDataset(
        data_dir=Path(cfg.data_dir)
    ).provide_dataset()

    train_dataset = TrainDataset(
        crop_sz=64,
        burst_size=cfg.burst_size,
    ).transform_dataset(zrr_dataset)

    # test_dataset = TestDataset(data_dir=Path(cfg.data_dir)).provide_dataset_pipeline()
    # test_dataset = TestDataset(data_dir=Path(cfg.data_dir)).provide_dataset()
    
    # NOTE: TorchTrainer does not support passing DatasetPipelines: https://docs.ray.io/en/master/ray-air/check-ingest.html#how-do-i-pass-in-a-datasetpipeline-to-my-trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=cfg.__dict__,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        dataset_config={
            "train": DatasetConfig(use_stream_api=True, stream_window_size=32),
        },
        datasets={"train": train_dataset},
    )
    result = trainer.fit()


    # auto_lr_find=True?
    # trainer = Trainer.from_argparse_args(
    #     args,
    #     deterministic=False,
    #     benchmark=True,
    #     # strategy=strategy,
    #     # strategy=BaguaStrategy(),
    #     callbacks=[
    #         # SWA
    #         StochasticWeightAveraging(swa_lrs=1e-2),
    #         # Larger min_delta because they are larger numbers
    #         EarlyStopping(
    #             "mse_L1", mode="max", min_delta=1e-2, verbose=True, patience=10
    #         ),
    #         EarlyStopping("psnr", mode="max", min_delta=1e-2, verbose=True, patience=5),
    #         # Between zero and one so we use a smaller min_delta
    #         EarlyStopping("ssim", mode="max", min_delta=1e-4, verbose=True, patience=5),
    #         EarlyStopping(
    #             "lpips", mode="min", min_delta=1e-4, verbose=True, patience=5
    #         ),
    #     ],
    # )
    # trainer.fit(_model, datamodule=data_module)


def train_loop_per_worker(cfg: dict[str, Any]):
    _cfg = Config.from_dict(cfg)
    dataset_shard: DatasetPipeline[TrainData] = session.get_dataset_shard("train") # type: ignore
    model = bsrt.make_model(_cfg)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model = train.torch.prepare_model(model)

    for epoch in range(2):
        for batches in dataset_shard.iter_torch_batches(
            batch_size=32
        ):
            print(f"Epoch {epoch} batch {batches}")
            break
            inputs, labels = torch.unsqueeze(batches["x"], 1), batches["y"]
            output = model(inputs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss.item()}")

        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.state_dict())
            ),
        )


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

    ################## model ##################
    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--act", type=str, default="relu", help="activation function"
    )
    model_group.add_argument(
        "--res_scale", type=float, default=1, help="residual scaling"
    )
    model_group.add_argument(
        "--shift_mean",
        type=bool,
        default=True,
        help="subtract pixel mean from the input",
    )
    model_group.add_argument(
        "--dilation", action="store_true", help="use dilated convolution"
    )

    ################## rcan ##################
    rcan_group = parser.add_argument_group("rcan")
    rcan_group.add_argument(
        "--n_resgroups", type=int, default=20, help="number of residual groups"
    )
    rcan_group.add_argument(
        "--reduction", type=int, default=16, help="number of feature maps reduction"
    )
    rcan_group.add_argument("--DA", action="store_true", help="use Dual Attention")
    rcan_group.add_argument("--CA", action="store_true", help="use Channel Attention")
    rcan_group.add_argument(
        "--non_local", action="store_true", help="use Dual Attention"
    )

    ################## training ##################
    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--batch_size", type=int, default=8, help="input batch size for training"
    )
    train_group.add_argument(
        "--gan_k", type=int, default=1, help="k value for adversarial loss"
    )

    ################## optimization ##################
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--decay", type=str, default="40-80", help="learning rate decay type"
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
        "--betas", type=tuple, default=(0.9, 0.999), help="ADAM beta"
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
    optimization_group.add_argument(
        "--gclip",
        type=float,
        default=0,
        help="gradient clipping threshold (0 = no clipping)",
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

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    cfg = Config.from_dict(args.__dict__)

    config = {
        "cfg": cfg,
        "args": args,
        # "layer_1": tune.choice([32, 64, 128]),
        # "layer_2": tune.choice([64, 128, 256]),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([32, 64, 128]),
    }

    train(config)

    # Make sure to pass in ``resources_per_trial`` using the ``get_tune_resources`` utility.
    # analysis = tune.run(
    #         train,
    #         metric="loss",
    #         mode="min",
    #         config=config,
    #         num_samples=1,
    #         resources_per_trial=get_tune_resources(num_workers=1),
    #         name="tune_mnist")

    # print("Best hyperparameters found were: ", analysis.best_config)
