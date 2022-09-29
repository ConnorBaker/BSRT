from argparse import ArgumentParser
from datasets.synthetic_burst.train_dataset import TrainData, TrainDataProcessor
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset
from option import Config, ConfigHyperTuner
from pandas import DataFrame
from pathlib import Path
from ray.air import session
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.air.config import CheckpointConfig, ScalingConfig, DatasetConfig, RunConfig
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.preprocessors.batch_mapper import BatchMapper
from ray.data.preprocessors.chain import Chain
from ray.train.torch import (
    TorchTrainer,
    prepare_model,
    prepare_optimizer,
    accelerate,
    backward,
)
from ray.tune import CLIReporter
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter
from ray.tune.search.optuna.optuna_search import OptunaSearch
from ray.tune.stopper import TrialPlateauStopper, CombinedStopper
from ray.tune.syncer import SyncConfig
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics.metric import Metric
from typing import Any
from utility import make_optimizer, make_model, make_loss_fn, make_psnr_fn
import logging
import optuna
import os
import ray
import torch
import torch.backends.cuda
import torch.backends.cudnn


def train_setup(cfg: Config):
    # Create the Tune Reporting Callback
    # session.report({"loss": 1.0}, checkpoint=Checkpoint.from_directory("/Users/connorbaker/Packages/BSRT/bsrt/checkpoints"))
    # metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    # trc = TuneReportCallback(metrics, on="validation_end")
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        run_config=RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project="bsrt",
                    api_key=os.environ["WANDB_API_KEY"],
                    save_checkpoints=True,
                    log_config=True,
                )
            ],
            stop=CombinedStopper(
                TrialPlateauStopper(
                    metric="metric.loss", std=0.005, num_results=8, grace_period=10
                ),
                TrialPlateauStopper(
                    metric="metric.psnr", std=0.01, num_results=8, grace_period=10
                ),
                TrialPlateauStopper(
                    metric="metric.ssim", std=0.01, num_results=8, grace_period=10
                ),
                TrialPlateauStopper(
                    metric="metric.lpips", std=0.01, num_results=8, grace_period=10
                ),
            ),
            verbose=1,
            progress_reporter=CLIReporter(
                metric_columns=[
                    "metric.loss",
                    "metric.psnr",
                    "metric.ssim",
                    "metric.lpips",
                ],
                print_intermediate_tables=True,
                sort_by_metric=True,
            ),
            log_to_file=True,
            checkpoint_config=CheckpointConfig(),
            sync_config=SyncConfig(upload_dir="gs://bsrt-supplemental"),
        ),
        scaling_config=ScalingConfig(num_workers=3),
        dataset_config={
            "train": DatasetConfig(use_stream_api=True),
        },
        datasets={
            "train": ZurichRaw2RgbDataset(data_dir=Path(cfg.data_dir))
            .provide_dataset()
            .limit(10)
        },
        preprocessor=Chain(
            BatchMapper(lambda df: df.drop("label", axis="columns")),
            TrainDataProcessor(crop_sz=256, burst_size=cfg.burst_size),
        ),
    )
    tuner = Tuner(
        trainable=trainer,
        tune_config=TuneConfig(
            metric="metric.loss",
            mode="min",
            search_alg=ConcurrencyLimiter(
                OptunaSearch(
                    metric=[
                        "metric.loss",
                        "metric.psnr",
                        "metric.ssim",
                        "metric.lpips",
                    ],
                    mode=["min", "max", "max", "min"],
                    sampler=optuna.samplers.NSGAIISampler(),
                ),
                max_concurrent=4,
            ),
            scheduler=AsyncHyperBandScheduler(
                time_attr="training_iteration",
                metric="metric.loss",
                mode="min",
                grace_period=10,
            ),
            num_samples=100,
            max_concurrent_trials=None,
            time_budget_s=None,
        ),
        param_space={
            "train_loop_config": cfg.__dict__ | ConfigHyperTuner().__dict__,
        },
    )
    analysis = tuner.fit()


def train_loop_per_worker(config: dict[str, Any]) -> None:
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    accelerate(amp=True)

    _cfg = Config(**config)
    data_shard: DatasetPipeline[TrainData] = session.get_dataset_shard("train")
    model: Module = prepare_model(make_model(_cfg))
    optimizer: Optimizer = prepare_optimizer(make_optimizer(_cfg, model))
    loss_fn: Metric = make_loss_fn(_cfg.loss, _cfg.data_type)
    psnr_fn: Metric = make_psnr_fn(_cfg.data_type)
    # model: BSRT = prepare_model(model, ddp_kwargs={"find_unused_parameters": True})

    # Use iter_epochs(10) to iterate over 10 epochs of data.
    for epoch_idx, epoch in enumerate(data_shard.iter_epochs(10)):
        for batch_idx, batch in enumerate(
            epoch.iter_batches(batch_size=cfg.batch_size)
        ):
            assert isinstance(batch, DataFrame)
            bursts = torch.stack(batch["burst"].tolist())
            gts = torch.stack(batch["gt"].tolist())

            # NOTE: while values should be in the range [0, 1], they are not clipped when training, so it is frequently the case that sr.max() > 1 or sr.min() < 0.
            optimizer.zero_grad()
            srs = model(bursts)

            loss: torch.Tensor = loss_fn(srs, gts)
            psnr_score, ssim_score, lpips_score = psnr_fn(srs, gts)

            backward(loss)
            optimizer.step()
            optimizer.schedule()

            session.report(
                {
                    "batch.number": batch_idx,
                    "batch.length": bursts.size(0),
                    "epoch.number": epoch_idx,
                    "metric.loss": loss.item(),
                    "metric.psnr": psnr_score.item(),
                    "metric.ssim": ssim_score.item(),
                    "metric.lpips": lpips_score.item(),
                },
                # checkpoint=Checkpoint.from_dict(
                #     {"epoch": epoch, "model": model.state_dict()}
                # ),
            )

    # View the stats for performance debugging.
    print(data_shard.stats())


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
    ray.init(
        configure_logging=True,
        logging_level=logging.INFO,
        object_store_memory=16 * 1024 * 1024 * 1024,
    )
    train_setup(cfg)
