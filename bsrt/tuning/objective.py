import os
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import torch
from lightning_lite.utilities.seed import seed_everything
from mfsr_utils.datasets.zurich_raw2rgb import ZurichRaw2Rgb
from mfsr_utils.pipelines.synthetic_burst_generator import (
    SyntheticBurstGeneratorData,
    SyntheticBurstGeneratorTransform,
)
from pytorch_lightning.callbacks import Callback, StochasticWeightAveraging
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer import Trainer
from syne_tune.constants import ST_CHECKPOINT_DIR  # type: ignore
from syne_tune.report import Reporter
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from typing_extensions import Literal

from bsrt.lightning_bsrt import LightningBSRT
from bsrt.tuning.cli_parser import (
    CLI_PARSER,
    PRECISION_MAP,
    OptimizerName,
    PrecisionName,
    SchedulerName,
    TunerConfig,
)
from bsrt.tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from bsrt.tuning.lr_scheduler.exponential_lr import ExponentialLRParams
from bsrt.tuning.lr_scheduler.one_cycle_lr import OneCycleLRParams
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
from bsrt.tuning.lr_scheduler.utilities import SchedulerParams, add_schedulers_to_argparse
from bsrt.tuning.model.bsrt import BSRTParams
from bsrt.tuning.optimizer.adamw import AdamWParams
from bsrt.tuning.optimizer.sgd import SGDParams
from bsrt.tuning.optimizer.utilities import OptimizerParams, add_optimizers_to_argparse
from bsrt.tuning.syne_tune_reporter_callback import SyneTuneReporterCallback


def get_optimizer_params(optimizer_name: OptimizerName, args: Namespace) -> OptimizerParams:
    if optimizer_name == "AdamW":
        return AdamWParams.from_args(args)
    elif optimizer_name == "SGD":
        return SGDParams.from_args(args)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


def get_scheduler_params(scheduler_name: SchedulerName, args: Namespace) -> SchedulerParams:
    if scheduler_name == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestartsParams.from_args(args)
    elif scheduler_name == "ExponentialLR":
        return ExponentialLRParams.from_args(args)
    elif scheduler_name == "OneCycleLR":
        return OneCycleLRParams.from_args(args)
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateauParams.from_args(args)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")


def get_lightning_precision(precision_name: PrecisionName) -> Literal["bf16", 16, 32, 64]:
    if precision_name == "bfloat16":
        return "bf16"
    elif precision_name == "float16":
        return 16
    elif precision_name == "float32":
        return 32
    elif precision_name == "float64":
        return 64
    else:
        raise ValueError(f"Precision {precision_name} not supported.")


def get_strategy() -> Strategy:
    return DDPStrategy(
        static_graph=True,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        ddp_comm_state=post_localSGD.PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=8,
        ),
        ddp_comm_hook=post_localSGD.post_localSGD_hook,  # type: ignore
        ddp_comm_wrapper=default_hooks.bf16_compress_wrapper,  # type: ignore
        model_averaging_period=4,
    )


def get_callbacks(st_checkpoint_dir: Optional[str] = None) -> List[Callback]:
    return [
        StochasticWeightAveraging(swa_lrs=1e-2),
        ModelCheckpoint(
            monitor="val/lpips",
            mode="min",
            auto_insert_metric_name=False,
            dirpath=st_checkpoint_dir,
            save_last=True,
        ),
        SyneTuneReporterCallback(
            metric_names=["val/psnr", "val/ms_ssim", "val/lpips"],
            reporter=Reporter(),
        ),
    ]


def objective(
    st_checkpoint_dir: str,
    tuner_config: TunerConfig,
    bsrt_params: BSRTParams,
    optimizer_params: OptimizerParams,
    scheduler_params: SchedulerParams,
) -> None:
    seed_everything(42)

    # Desired batch size
    target_batch_size: int = 64

    # Number of batches a single GPU can handle in memory
    single_gpu_batch_size: int = tuner_config.batch_size

    # Number of GPUs
    import torch.cuda

    num_gpus: int = torch.cuda.device_count()

    # Number of batches to accumulate before performing a backward pass
    actual_batch_size: int = single_gpu_batch_size * num_gpus

    # Number of batches to accumulate before performing a backward pass
    accumulate_batch_size: int = target_batch_size // actual_batch_size

    # Num CPUs
    num_cpus: None | int = os.cpu_count()
    assert num_cpus is not None

    # TODO: Add name/version to make it clear we're resuming runs
    wandb_logger = WandbLogger(
        entity="connorbaker", project="bsrt", group=tuner_config.experiment_name, reinit=True
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=get_lightning_precision(tuner_config.precision),
        enable_checkpointing=True,
        limit_train_batches=tuner_config.limit_train_batches,
        limit_val_batches=tuner_config.limit_val_batches,
        max_epochs=tuner_config.max_epochs,
        # strategy=get_strategy(),
        detect_anomaly=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accumulate_batch_size,
        callbacks=get_callbacks(st_checkpoint_dir),
    )

    model = LightningBSRT(
        bsrt_params=bsrt_params,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
    )

    hyperparams = {
        f"{params.__class__.__name__}/{k}": v
        for params in (optimizer_params, scheduler_params, bsrt_params)
        for k, v in params.__dict__.items()
    }

    data_dir: Path = Path(tuner_config.data_dir)
    full_dataset: ZurichRaw2Rgb[SyntheticBurstGeneratorData] = ZurichRaw2Rgb(
        data_dir,
        transform=SyntheticBurstGeneratorTransform(
            burst_size=14, crop_sz=256, dtype=PRECISION_MAP[tuner_config.precision]
        ),
    )
    train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2])
    train_data_loader: DataLoader[SyntheticBurstGeneratorData] = DataLoader(
        train_dataset,
        batch_size=single_gpu_batch_size,
        num_workers=num_cpus,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_data_loader: DataLoader[SyntheticBurstGeneratorData] = DataLoader(
        val_dataset,
        batch_size=single_gpu_batch_size,
        num_workers=num_cpus,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    for _logger in trainer.loggers:
        _logger.log_hyperparams(hyperparams)

    try:
        ckpt_path = Path(st_checkpoint_dir) / "last.ckpt"
        if ckpt_path.exists():
            print(f"Loading checkpoint from {ckpt_path}")
        else:
            print(f"No checkpoint found at {ckpt_path}")
        trainer.fit(  # type: ignore
            model=model,
            train_dataloaders=train_data_loader,
            val_dataloaders=val_data_loader,
            ckpt_path=ckpt_path.as_posix() if ckpt_path.exists() else None,
        )

        wandb_logger.experiment.finish()  # type: ignore
    except Exception as e:
        wandb_logger.experiment.finish(1)  # type: ignore
        raise e


if __name__ == "__main__":
    parser = CLI_PARSER
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str, required=True)

    BSRTParams.add_to_argparse(parser)
    add_optimizers_to_argparse(parser)
    add_schedulers_to_argparse(parser)

    args = parser.parse_args()

    tuner_config = TunerConfig.from_args(args)
    bsrt_params = BSRTParams.from_args(args)
    optimizer_params = get_optimizer_params(args.optimizer, args)
    scheduler_params = get_scheduler_params(args.scheduler, args)

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

    import torch.backends.cuda
    import torch.backends.cudnn

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")  # type: ignore

    objective(
        args.st_checkpoint_dir,
        tuner_config,
        bsrt_params,
        optimizer_params,
        scheduler_params,
    )
