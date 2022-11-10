import sys
from logging import StreamHandler
from typing import Mapping, NewType, Tuple

import wandb
from lightning_lite.utilities.seed import seed_everything
from optuna import Trial
from optuna.exceptions import TrialPruned
from optuna.logging import get_logger
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from typing_extensions import Literal

from bsrt.lighting_bsrt import LightningBSRT
from bsrt.tuning.cli_parser import PrecisionName, TunerConfig
from bsrt.tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from bsrt.tuning.lr_scheduler.exponential_lr import ExponentialLRParams
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
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


class MinEpochsEarlyStopping(EarlyStopping):
    """
    A custom early stopping callback that doesn't stop training if the minimum number of epochs
    hasn't been reached yet. This is useful for when we want to train for a longer time to see if
    the model can recover from a bad initialization.

    Patience is incremented by 1 if the minimum number of epochs hasn't been reached yet, ensuring
    that we don't stop training prematurely. It is reset to the original value once the minimum
    number of epochs has been reached. In this way, patience does not start until we have reached
    the minimum number of epochs.

    When the minimum number of epochs has been reached, in addition to patience being reset to its
    original value, the divergence threshold is added to the callback and the wait counter is
    reset to zero.
    """

    def __init__(
        self,
        monitor: str,
        min_delta: float,
        patience: int,
        min_epochs: int,
        mode: Literal["min", "max"],
        divergence_threshold: float,
        verbose: bool = False,
    ):
        """
        Args:
            divergence_threshold (float): The divergence threshold for the metric to be monitored.
            monitor (str): The metric to be monitored.
            patience (int): The number of epochs to wait for the metric to be monitored to improve.
            min_epochs (int): The minimum number of epochs to train for.
            min_delta (float): The minimum change in the monitored metric to qualify as an
                improvement.
            mode (Literal["min", "max"]): Whether the monitored metric should be increasing or
                decreasing.
            verbose (bool, optional): Whether to print messages. Defaults to False.
        """
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            verbose=verbose,
        )
        self.min_epochs = min_epochs
        self._divergence_threshold = divergence_threshold

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch < self.min_epochs:
            # Reset the wait counter to 0
            self.wait_count = 0
        elif trainer.current_epoch == self.min_epochs:
            # Reset the wait counter to 0
            self.wait_count = 0
            # Add the divergence threshold attribute now that we've reached the minimum number of
            # epochs
            self.divergence_threshold = self._divergence_threshold

        self._run_early_stopping_check(trainer)


def objective(
    config: TunerConfig,
    datamodule: LightningDataModule,
    trial: Trial,
) -> Tuple[PSNR, MS_SSIM, LPIPS]:
    seed_everything(42)

    if config.optimizer == "AdamW":
        optimizer_params = AdamWParams.suggest(trial)
    elif config.optimizer == "SGD":
        optimizer_params = SGDParams.suggest(trial)
    else:
        error_msg = f"Optimizer {config.optimizer} not supported."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if config.scheduler == "ReduceLROnPlateau":
        scheduler_params = ReduceLROnPlateauParams.suggest(trial)
    elif config.scheduler == "CosineAnnealingWarmRestarts":
        scheduler_params = CosineAnnealingWarmRestartsParams.suggest(trial)
    elif config.scheduler == "ExponentialLR":
        scheduler_params = ExponentialLRParams.suggest(trial)
    else:
        error_msg = f"Scheduler {config.scheduler} not supported."
        logger.error(error_msg)
        raise ValueError(error_msg)

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

    wandb_kwargs = {
        "entity": "connorbaker",
        "project": "bsrt",
        # "group": Provided by the experiment name passed in from the command line
        "reinit": True,
        "settings": wandb.Settings(start_method="fork"),
    }

    wandb_kwargs["group"] = config.experiment_name
    wandb_logger = WandbLogger(**wandb_kwargs)

    precision_name_to_lightning_precision: Mapping[PrecisionName, Literal["bf16", 16, 32, 64]] = {
        "bfloat16": "bf16",
        "float16": 16,
        "float32": 32,
        "float64": 64,
    }

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=precision_name_to_lightning_precision[config.precision],
        enable_checkpointing=False,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        max_epochs=config.max_epochs,
        detect_anomaly=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=wandb_logger,
        replace_sampler_ddp=False,
        num_sanity_val_steps=0,
        callbacks=[
            MinEpochsEarlyStopping(
                monitor="val/psnr",
                min_delta=1.0,
                patience=5,
                mode="max",
                divergence_threshold=PSNR_DIVERGENCE_THRESHOLD,
                min_epochs=10,
                verbose=True,
            ),
            MinEpochsEarlyStopping(
                monitor="val/ms_ssim",
                min_delta=0.1,
                patience=5,
                mode="max",
                divergence_threshold=MS_SSIM_DIVERGENCE_THRESHOLD,
                min_epochs=10,
                verbose=True,
            ),
            MinEpochsEarlyStopping(
                monitor="val/lpips",
                min_delta=0.1,
                patience=5,
                mode="min",
                divergence_threshold=LPIPS_DIVERGENCE_THRESHOLD,
                min_epochs=10,
                verbose=True,
            ),
        ],
    )

    for _logger in trainer.loggers:
        _logger.log_hyperparams(hyperparams)

        if isinstance(_logger, WandbLogger):
            _logger.watch(model, log="all", log_graph=True)

    try:
        trainer.fit(model=model, datamodule=datamodule)

        psnr = PSNR(trainer.callback_metrics["val/psnr"].item())
        ms_ssim = MS_SSIM(trainer.callback_metrics["val/ms_ssim"].item())
        lpips = LPIPS(trainer.callback_metrics["val/lpips"].item())

        logger.info(f"Finihsed training with PSNR: {psnr}, MS-SSIM: {ms_ssim}, LPIPS: {lpips}")
        wandb.finish()
        return psnr, ms_ssim, lpips
    except Exception as e:
        if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory error: {e}")
            wandb.finish(1)
            raise TrialPruned()
        # TODO: We want to make sure this is a known value error (e.g. NAN).
        # elif isinstance(e, ValueError):
        #     logger.error(f"Value error: {e}")
        #     wandb.finish(1)
        #     raise TrialPruned()
        else:
            logger.error(f"Unknown error: {e}")
            wandb.finish(1)
            raise e
