import sys
from logging import StreamHandler
from typing import List, NewType, Tuple, Union

from lightning_lite.utilities.seed import seed_everything
from optuna import Trial
from optuna.exceptions import TrialPruned
from optuna.logging import get_logger
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer import Trainer
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD
from typing_extensions import Literal

from bsrt.lighting_bsrt import LightningBSRT
from bsrt.tuning.cli_parser import OptimizerName, PrecisionName, TunerConfig
from bsrt.tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from bsrt.tuning.lr_scheduler.exponential_lr import ExponentialLRParams
from bsrt.tuning.lr_scheduler.one_cycle_lr import OneCycleLRParams
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
from bsrt.tuning.min_epochs_early_stopping import MinEpochsEarlyStopping
from bsrt.tuning.model.bsrt import BSRTParams
from bsrt.tuning.optimizer.adamw import AdamWParams
from bsrt.tuning.optimizer.sgd import SGDParams

logger = get_logger("bsrt.tuning.objective")
logger.addHandler(StreamHandler(sys.stdout))

# Create a type to model the different training errors we can recieve

PSNR = NewType("PSNR", float)
MS_SSIM = NewType("MS_SSIM", float)
LPIPS = NewType("LPIPS", float)

PSNR_DIVERGENCE_THRESHOLD: float = 16.0
MS_SSIM_DIVERGENCE_THRESHOLD: float = 0.8
LPIPS_DIVERGENCE_THRESHOLD: float = 0.2


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
        error_msg = f"Precision {precision_name} not supported."
        logger.error(error_msg)
        raise ValueError(error_msg)


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


def get_callbacks() -> List[Callback]:
    return [
        # StochasticWeightAveraging(swa_lrs=1e-3),
        ModelCheckpoint(
            monitor="val/lpips",
            mode="min",
            auto_insert_metric_name=False,
        ),
        MinEpochsEarlyStopping(
            monitor="val/psnr",
            min_delta=1.0,
            patience=3,
            mode="max",
            divergence_threshold=PSNR_DIVERGENCE_THRESHOLD / 2,
            min_epochs=5,
            verbose=True,
        ),
        MinEpochsEarlyStopping(
            monitor="val/ms_ssim",
            min_delta=0.1,
            patience=3,
            mode="max",
            divergence_threshold=MS_SSIM_DIVERGENCE_THRESHOLD / 2,
            min_epochs=5,
            verbose=True,
        ),
        MinEpochsEarlyStopping(
            monitor="val/lpips",
            min_delta=0.1,
            patience=3,
            mode="min",
            divergence_threshold=LPIPS_DIVERGENCE_THRESHOLD * 2,
            min_epochs=5,
            verbose=True,
        ),
    ]


def get_optimizer_params(
    optimizer_name: OptimizerName, trial: Trial
) -> Union[AdamWParams, SGDParams]:
    if optimizer_name == "AdamW":
        return AdamWParams.suggest(trial)
    elif optimizer_name == "SGD":
        return SGDParams.suggest(trial)
    else:
        error_msg = f"Optimizer {optimizer_name} not supported."
        logger.error(error_msg)
        raise ValueError(error_msg)


def get_lr_scheduler_params(
    lr_scheduler_name: str, trial: Trial, max_epochs: int, batch_size: int
) -> Union[
    CosineAnnealingWarmRestartsParams,
    ExponentialLRParams,
    OneCycleLRParams,
    ReduceLROnPlateauParams,
]:
    if lr_scheduler_name == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestartsParams.suggest(trial)
    elif lr_scheduler_name == "ExponentialLR":
        return ExponentialLRParams.suggest(trial)
    elif lr_scheduler_name == "OneCycleLR":
        return OneCycleLRParams.suggest(trial, max_epochs, batch_size)
    elif lr_scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateauParams.suggest(trial)
    else:
        error_msg = f"Learning rate scheduler {lr_scheduler_name} not supported."
        logger.error(error_msg)
        raise ValueError(error_msg)


def objective(
    config: TunerConfig,
    datamodule: LightningDataModule,
    trial: Trial,
) -> Tuple[PSNR, MS_SSIM, LPIPS]:
    seed_everything(42)

    optimizer_params = get_optimizer_params(config.optimizer, trial)
    scheduler_params = get_lr_scheduler_params(
        config.scheduler, trial, config.max_epochs, config.batch_size
    )
    bsrt_params = BSRTParams.suggest(trial)

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

    wandb_logger = WandbLogger(
        entity="connorbaker", project="bsrt", group=config.experiment_name, reinit=True
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=get_lightning_precision(config.precision),
        enable_checkpointing=True,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        max_epochs=config.max_epochs,
        strategy=get_strategy(),
        detect_anomaly=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=get_callbacks(),
    )

    for _logger in trainer.loggers:
        _logger.log_hyperparams(hyperparams)

        # if isinstance(_logger, WandbLogger):
        #     _logger.watch(model, log="all", log_graph=True)

    try:
        trainer.fit(model=model, datamodule=datamodule)

        psnr = PSNR(trainer.callback_metrics["val/psnr"].item())
        ms_ssim = MS_SSIM(trainer.callback_metrics["val/ms_ssim"].item())
        lpips = LPIPS(trainer.callback_metrics["val/lpips"].item())

        logger.info(f"Finihsed training with PSNR: {psnr}, MS-SSIM: {ms_ssim}, LPIPS: {lpips}")
        wandb_logger.experiment.finish()
        return psnr, ms_ssim, lpips
    except Exception as e:
        wandb_logger.experiment.finish(1)
        if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory error: {e}")
            raise TrialPruned()
        elif isinstance(e, ValueError) and "tensor(nan" in str(e):
            logger.error(f"NAN value error: {e}")
            raise TrialPruned()
        else:
            logger.error(f"Unknown error: {e}")
            raise e
