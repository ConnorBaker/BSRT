from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union

import wandb
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.bagua import BaguaStrategy
from pytorch_lightning.trainer import Trainer

from bsrt.tuning.cli_parser import TunerConfig

from ..datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
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


# Create a type to model the different training errors we can recieve
class TrainingError(Enum):
    CUDA_OOM = "CUDA out of memory"
    FORWARD_RETURNED_NAN = "Forward pass returned NaN"
    MODEL_PARAMS_INVALID = "Model parameters are invalid"
    OPTIMIZER_PARAMS_INVALID = "Optimizer parameters are invalid"
    SCHEDULER_PARAMS_INVALID = "Scheduler parameters are invalid"
    UNKNOWN_OPTIMIZER = "Unknown optimizer"
    UNKNOWN_SCHEDULER = "Unknown scheduler"
    UNKNOWN_ERROR = "Unknown error"


@dataclass
class ObjectiveMetrics:
    psnr: float
    ms_ssim: float
    lpips: float


def objective(
    params: Dict[str, Any],
    wandb_kwargs: Dict[str, Union[str, bool]],
    config: TunerConfig,
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
            return TrainingError.UNKNOWN_OPTIMIZER
    except TypeError:
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
            return TrainingError.UNKNOWN_SCHEDULER

    except TypeError:
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
        return TrainingError.MODEL_PARAMS_INVALID

    # swa_lrs = trial.suggest_float("swa_lr", 1e-5, 1e-1, log=True)
    # hyperparameters["swa_lr"] = swa_lrs

    ### Setup the trainer ###
    # if cli.config_init.trainer.callbacks is None:
    #     cli.config_init.trainer.callbacks = []

    # cli.config_init.trainer.callbacks.append(swa)
    # cli.config_init.trainer.callbacks.extend(
    #     PyTorchLightningPruningCallback(trial, monitor=metric_name)
    #     for (metric_name, _) in metric_names_and_directions
    # )

    wandb_logger = WandbLogger(**wandb_kwargs)
    wandb_logger.log_hyperparams(params)
    wandb_logger.watch(model, log="all", log_graph=True)

    datamodule = SyntheticZurichRaw2RgbDataModule(
        precision=config.precision,
        crop_size=256,
        data_dir="/home/connorbaker/ramdisk/datasets",
        burst_size=14,
        batch_size=16,
        num_workers=-1,
        pin_memory=True,
        persistent_workers=True,
        cache_in_gb=40,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=config.precision,
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
            EarlyStopping("train/psnr", min_delta=1.0, patience=10, mode="max"),
            EarlyStopping("train/ms_ssim", min_delta=0.1, patience=10, mode="max"),
            EarlyStopping("train/lpips", min_delta=0.1, patience=10, mode="min"),
        ],
    )

    try:
        trainer.fit(model=model, datamodule=datamodule)
        wandb.finish()
        return ObjectiveMetrics(
            psnr=trainer.callback_metrics["train/psnr"].item(),
            ms_ssim=trainer.callback_metrics["train/ms_ssim"].item(),
            lpips=trainer.callback_metrics["train/lpips"].item(),
        )
    except Exception as e:  # Out of memory
        wandb.finish(1)
        if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
            return TrainingError.CUDA_OOM
        elif isinstance(e, ValueError):
            return TrainingError.FORWARD_RETURNED_NAN
        else:
            return TrainingError.UNKNOWN_ERROR
