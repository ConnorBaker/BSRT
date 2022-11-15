from pathlib import Path
from typing import List, Optional

from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer import Trainer
from syne_tune.report import Reporter
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD
from typing_extensions import Literal

from bsrt.datasets.synthetic_zurich_raw2rgb import SyntheticZurichRaw2Rgb
from bsrt.lighting_bsrt import LightningBSRT
from bsrt.tuning.cli_parser import CLI_PARSER, PRECISION_MAP, PrecisionName, TunerConfig
from bsrt.tuning.lr_scheduler.one_cycle_lr import OneCycleLRParams
from bsrt.tuning.lr_scheduler.utilities import SchedulerParams
from bsrt.tuning.model.bsrt import BSRTParams
from bsrt.tuning.optimizer.adamw import AdamWParams
from bsrt.tuning.optimizer.utilities import OptimizerParams
from bsrt.tuning.syne_tune_reporter_callback import SyneTuneReporterCallback


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
        ddp_comm_hook=post_localSGD.post_localSGD_hook,
        ddp_comm_wrapper=default_hooks.bf16_compress_wrapper,
        model_averaging_period=4,
    )


def get_callbacks(st_checkpoint_dir: Optional[str] = None) -> List[Callback]:
    return [
        # StochasticWeightAveraging(swa_lrs=1e-3),
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
    data_module: LightningDataModule,
    bsrt_params: BSRTParams,
    optimizer_params: OptimizerParams,
    scheduler_params: SchedulerParams,
) -> None:
    seed_everything(42)

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
        strategy=get_strategy(),
        detect_anomaly=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=get_callbacks(st_checkpoint_dir),
    )

    for _logger in trainer.loggers:
        _logger.log_hyperparams(hyperparams)

        # if isinstance(_logger, WandbLogger):
        #     _logger.watch(model, log="all", log_graph=True)

    try:
        ckpt_path = Path(st_checkpoint_dir) / "last.ckpt"
        if ckpt_path.exists():
            print(f"Loading checkpoint from {ckpt_path}")
        else:
            print(f"No checkpoint found at {ckpt_path}")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path.as_posix() if ckpt_path.exists() else None,
        )

        wandb_logger.experiment.finish()
    except Exception as e:
        wandb_logger.experiment.finish(1)
        raise e


if __name__ == "__main__":
    parser = CLI_PARSER
    parser.add_argument("--st_checkpoint_dir", type=str, required=True)

    BSRTParams.add_to_argparse(parser)
    OneCycleLRParams.add_to_argparse(parser)
    AdamWParams.add_to_argparse(parser)

    args = parser.parse_args()

    tuner_config = TunerConfig.from_args(args)
    bsrt_params = BSRTParams.from_args(args)
    optimizer_params = AdamWParams.from_args(args)
    scheduler_params = OneCycleLRParams.from_args(args)

    datamodule = SyntheticZurichRaw2Rgb(
        precision=PRECISION_MAP[tuner_config.precision],
        crop_size=256,
        data_dir=tuner_config.data_dir,
        burst_size=14,
        batch_size=tuner_config.batch_size,
        num_workers=-1,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    objective(
        args.st_checkpoint_dir,
        tuner_config,
        datamodule,
        bsrt_params,
        optimizer_params,
        scheduler_params,
    )
