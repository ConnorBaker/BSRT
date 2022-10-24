import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union

import wandb
from ax.utils.common.logger import build_stream_handler, get_logger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.bagua import BaguaStrategy
from pytorch_lightning.trainer import Trainer
from typing_extensions import Literal

from bsrt.tuning.cli_parser import TunerConfig

from ..lighting_bsrt import LightningBSRT
from .lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from .lr_scheduler.exponential_lr import ExponentialLRParams
from .lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
from .model.bsrt import BSRTParams
from .optimizer.decoupled_adamw import DecoupledAdamWParams
from .optimizer.decoupled_sgdw import DecoupledSGDWParams
from .utilities import filter_and_remove_from_keys

logger = get_logger("bsrt.tuning.objective")
logger.addHandler(build_stream_handler())


# Create a type to model the different training errors we can recieve
class TrainingError(Enum):
    CUDA_OOM = "CUDA out of memory"
    FORWARD_RETURNED_NAN = "Forward pass returned NaN"
    MODEL_PARAMS_INVALID = "Model parameters are invalid"
    OPTIMIZER_PARAMS_INVALID = "Optimizer parameters are invalid"
    SCHEDULER_PARAMS_INVALID = "Scheduler parameters are invalid"
    UNKNOWN_OPTIMIZER = "Unknown optimizer"
    UNKNOWN_SCHEDULER = "Unknown scheduler"


@dataclass
class ObjectiveMetrics:
    psnr: float
    ms_ssim: float
    lpips: float


PSNR_DIVERGENCE_THRESHOLD: float = 16.0
MS_SSIM_DIVERGENCE_THRESHOLD: float = 0.8
LPIPS_DIVERGENCE_THRESHOLD: float = 0.2


class MinEpochsEarlyStopping(EarlyStopping):
    """
    A custom early stopping callback that doesn't stop training if the minimum number of epochs hasn't been reached yet. This is useful for when we want to train for a longer time to see if the model can recover from a bad initialization.

    Patience is incremented by 1 if the minimum number of epochs hasn't been reached yet, ensuring that we don't stop training prematurely. It is reset to the original value once the minimum number of epochs has been reached. In this way, patience does not start until we have reached the minimum number of epochs.

    When the minimum number of epochs has been reached, in addition to patience being reset to its original value, the divergence threshold is added to the callback and the wait counter is reset to zero.
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
            min_delta (float): The minimum change in the monitored metric to qualify as an improvement.
            mode (Literal["min", "max"]): Whether the monitored metric should be increasing or decreasing.
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

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch < self.min_epochs:
            # Reset the wait counter to 0
            self.wait_count = 0
        elif trainer.current_epoch == self.min_epochs:
            # Reset the wait counter to 0
            self.wait_count = 0
            # Add the divergence threshold attribute now that we've reached the minimum number of epochs
            self.divergence_threshold = self._divergence_threshold

        self._run_early_stopping_check(trainer)


def objective(
    params: Dict[str, Any],
    config: TunerConfig,
    datamodule: LightningDataModule,
) -> Union[TrainingError, ObjectiveMetrics]:
    seed_everything(42)
    try:
        if any(k.startswith("decoupled_adamw_params") for k in params):
            params["decoupled_adamw_params.betas"] = (
                params.pop("decoupled_adamw_params.beta_gradient"),
                params.pop("decoupled_adamw_params.beta_square"),
            )
            optimizer_params = DecoupledAdamWParams(
                **filter_and_remove_from_keys("decoupled_adamw_params", params)
            )
        elif any(k.startswith("decoupled_sgdw_params") for k in params):
            optimizer_params = DecoupledSGDWParams(
                **filter_and_remove_from_keys("decoupled_sgdw_params", params)
            )
        else:
            logger.error(f"Unknown optimizer: {params}")
            return TrainingError.UNKNOWN_OPTIMIZER
    except TypeError:
        logger.error(f"Optimizer parameters are invalid: {params}")
        return TrainingError.OPTIMIZER_PARAMS_INVALID

    try:
        if any(k.startswith("cosine_annealing_warm_restarts_params") for k in params):
            scheduler_params = CosineAnnealingWarmRestartsParams(
                **filter_and_remove_from_keys(
                    "cosine_annealing_warm_restarts_params", params
                )
            )
        elif any(k.startswith("exponential_lr_params") for k in params):
            scheduler_params = ExponentialLRParams(
                **filter_and_remove_from_keys("exponential_lr_params", params)
            )
        elif any(k.startswith("reduce_lr_on_plateau_params") for k in params):
            scheduler_params = ReduceLROnPlateauParams(
                **filter_and_remove_from_keys("reduce_lr_on_plateau_params", params)
            )
        else:
            logger.error(f"Unknown scheduler: {params}")
            return TrainingError.UNKNOWN_SCHEDULER

    except TypeError:
        logger.error(f"Scheduler parameters are invalid: {params}")
        return TrainingError.SCHEDULER_PARAMS_INVALID

    try:
        bsrt_params = BSRTParams(**filter_and_remove_from_keys("bsrt_params", params))
        model = LightningBSRT(
            bsrt_params=bsrt_params,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            use_speed_opts=True,
            use_quality_opts=True,
        )
    except TypeError:
        logger.error(f"Model parameters are invalid: {params}")
        return TrainingError.MODEL_PARAMS_INVALID

    wandb_kwargs = {
        "entity": "connorbaker",
        "project": "bsrt",
        # "group": Provided by the experiment name passed in from the command line
        "reinit": True,
        "settings": wandb.Settings(start_method="fork"),
    }
    wandb.login(key=config.wandb_api_key)
    wandb_kwargs["group"] = config.experiment_name
    wandb_logger = WandbLogger(**wandb_kwargs)

    if config.precision == "bf16":
        precision = "bf16"
    elif config.precision == "16":
        precision = 16
    elif config.precision == "32":
        precision = 32
    else:
        logger.error(f"Unknown precision: {config.precision}")
        raise ValueError(f"Unknown precision {config.precision}")

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=precision,
        enable_checkpointing=False,
        strategy=BaguaStrategy(algorithm="gradient_allreduce"),
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        max_epochs=config.max_epochs,
        detect_anomaly=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=wandb_logger,
        replace_sampler_ddp=False,
        callbacks=[
            MinEpochsEarlyStopping(
                monitor="val/psnr",
                min_delta=0.5,
                patience=5,
                mode="max",
                divergence_threshold=PSNR_DIVERGENCE_THRESHOLD,
                min_epochs=10,
                verbose=True,
            ),
            MinEpochsEarlyStopping(
                monitor="val/ms_ssim",
                min_delta=0.05,
                patience=5,
                mode="max",
                divergence_threshold=MS_SSIM_DIVERGENCE_THRESHOLD,
                min_epochs=10,
                verbose=True,
            ),
            MinEpochsEarlyStopping(
                monitor="val/lpips",
                min_delta=0.05,
                patience=5,
                mode="min",
                divergence_threshold=LPIPS_DIVERGENCE_THRESHOLD,
                min_epochs=10,
                verbose=True,
            ),
        ],
    )

    for _logger in trainer.loggers:
        _logger.log_hyperparams(params)

        if isinstance(_logger, WandbLogger):
            _logger.watch(model, log="all", log_graph=True)

    try:
        trainer.fit(model=model, datamodule=datamodule)
        objective_metrics = ObjectiveMetrics(
            psnr=trainer.callback_metrics["val/psnr"].item(),
            ms_ssim=trainer.callback_metrics["val/ms_ssim"].item(),
            lpips=trainer.callback_metrics["val/lpips"].item(),
        )
        logger.info(f"Finished with objective metrics: {objective_metrics}")
        wandb.finish()
        return objective_metrics
    except Exception as e:
        if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory error: {e}")
            wandb.finish(1)
            return TrainingError.CUDA_OOM
        elif isinstance(e, ValueError):
            logger.error(f"Value error: {e}")
            wandb.finish(1)
            return TrainingError.FORWARD_RETURNED_NAN
        else:
            logger.error(f"Unknown error: {e}")
            wandb.finish(1)
            raise e
