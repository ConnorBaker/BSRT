from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import torch
import torch.nn.functional as F
import wandb
from ax import optimize
from ax.service.utils.instantiation import ObjectiveProperties
from bsrt.hyperparameter_tuning.optimizer.adam import AdamParams
from bsrt.hyperparameter_tuning.optimizer.sgd import SGDParams
from bsrt.lighting_bsrt import LightningBSRT
from data_processing.camera_pipeline import demosaic
from datasets.synthetic_burst.train_dataset import TrainData
from datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
from bsrt.model.bsrt import BSRT
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer import Trainer
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import (
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
from typing_extensions import Literal, TypedDict
import apex.optimizers as apex_optimizers
from ax.service.ax_client import AxClient
from hyperparameter_tuning.model.bsrt import BSRTParams
from utilities import filter_and_remove_from_keys

SchedulerName = Literal[
    "ExponentialLR",
    "ReduceLROnPlateau",
    "OneCycleLR",
    "CosineAnnealingWarmRestarts",
]

wandb_kwargs = {
    "entity": "connorbaker",
    "project": "bsrt",
    "group": "bf16-hyperparameter-tuning",
    "reinit": True,
    "settings": wandb.Settings(start_method="fork"),
}
# wandbc = WeightsAndBiasesCallback(
#     metric_name="final train loss",
#     wandb_kwargs=wandb_kwargs,
#     as_multirun=True,
# )

# Create a type to model the different training errors we can recieve
class TrainingError(Enum):
    CUDA_OOM = "CUDA out of memory"
    FORWARD_RETURNED_NAN = "Forward pass returned NaN"
    MODEL_PARAMS_INVALID = "Model parameters are invalid"
    OPTIMIZER_PARAMS_INVALID = "Optimizer parameters are invalid"
    UNKNOWN_OPTIMIZER = "Unknown optimizer"


@dataclass
class ObjectiveMetrics:
    psnr: float
    ms_ssim: float
    lpips: float


def objective(params: Dict[str, Any]) -> Union[TrainingError, ObjectiveMetrics]:
    # Update the existing configurations from the CLI
    # NOTE: When updating values which do not require initialization, use
    # cli.config. For classes like the logger or callbacks, use cli.
    # config_init because we are passing in a new, initialized instances of
    # the class, not the class path and arguments.

    try:
        if any(k.startswith("adam_params") for k in params):
            params["adam_params.betas"] = (
                params.pop("adam_params.beta_gradient"),
                params.pop("adam_params.beta_square"),
            )
            optimizer_params = AdamParams(
                **filter_and_remove_from_keys("adam_params", params)
            )
        elif any(k.startswith("sgd_params") for k in params):
            optimizer_params = SGDParams(
                **filter_and_remove_from_keys("sgd_params", params)
            )
        else:
            return TrainingError.UNKNOWN_OPTIMIZER
    except TypeError:
        return TrainingError.OPTIMIZER_PARAMS_INVALID

    try:
        bsrt_params = BSRTParams(**filter_and_remove_from_keys("model_params", params))
        model = LightningBSRT(
            bsrt_params=bsrt_params, optimizer_params=optimizer_params
        )
    except TypeError:
        return TrainingError.MODEL_PARAMS_INVALID

    # TODO: Other optimizers
    # apex_optimizers.FusedNovoGrad
    # apex_optimizers.FusedMixedPrecisionLamb()

    ### Scheduler Hyperparameters ###
    # scheduler_name: SchedulerName = cast(
    #     SchedulerName,
    #     trial.suggest_categorical("scheduler_name", get_args(SchedulerName)),
    # )
    # hyperparameters["scheduler_name"] = scheduler_name
    # match scheduler_name:
    #     case "ExponentialLR":
    #         gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    #         hyperparameters["gamma"] = gamma

    #         lr_scheduler = ExponentialLR(
    #             optimizer,
    #             gamma=gamma,
    #         )

    #     case "ReduceLROnPlateau":
    #         factor = trial.suggest_float("factor", 1e-4, 1.0, log=True)
    #         hyperparameters["factor"] = factor

    #         lr_scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=10)

    #     case "OneCycleLR":
    #         match optimizer_name:
    #             case "AdamW":
    #                 max_lr = trial.suggest_float("max_lr", adam_lr, 1e-1, log=True)
    #             case "SGD":
    #                 max_lr = trial.suggest_float("max_lr", sgd_lr, 1e-1, log=True)

    #         hyperparameters["max_lr"] = max_lr

    #         lr_scheduler = OneCycleLR(
    #             optimizer,
    #             max_lr=max_lr,
    #             steps_per_epoch=cli.config.data.batch_size,
    #             epochs=cli.config.trainer.max_epochs,
    #         )

    #     case "CosineAnnealingWarmRestarts":
    #         T_0 = trial.suggest_int("T_0", 1, 1000)
    #         hyperparameters["T_0"] = T_0

    #         T_mult = trial.suggest_int("T_mult", 1, 10)
    #         hyperparameters["T_mult"] = T_mult

    #         eta_min = trial.suggest_float("eta_min", 1e-10, 1e-3, log=True)
    #         hyperparameters["eta_min"] = eta_min

    #         lr_scheduler = CosineAnnealingWarmRestarts(
    #             optimizer,
    #             T_0=T_0,
    #             T_mult=T_mult,
    #             eta_min=eta_min,
    #         )

    # cli.config_init.lr_scheduler = lr_scheduler

    # swa_lrs = trial.suggest_float("swa_lr", 1e-5, 1e-1, log=True)
    # hyperparameters["swa_lr"] = swa_lrs
    # swa = StochasticWeightAveraging(swa_lrs=swa_lrs)

    ### Setup the trainer ###
    # if cli.config_init.trainer.callbacks is None:
    #     cli.config_init.trainer.callbacks = []

    # cli.config_init.trainer.callbacks.append(swa)
    # cli.config_init.trainer.callbacks.extend(
    #     PyTorchLightningPruningCallback(trial, monitor=metric_name)
    #     for (metric_name, _) in metric_names_and_directions
    # )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=16,
        enable_checkpointing=False,
        strategy=DDPStrategy(find_unused_parameters=False),
        limit_train_batches=100,
        limit_val_batches=10,
        max_epochs=2,
    )
    dm = SyntheticZurichRaw2RgbDataModule(
        precision=16,
        crop_size=256,
        data_dir="/home/connorbaker/ramdisk/datasets",
        burst_size=14,
        batch_size=32,
        num_workers=-1,
        pin_memory=True,
        persistent_workers=True,
    )
    # logger = WandbLogger(**wandb_kwargs)
    # logger.log_hyperparams(params)
    # logger.watch(model, log="all", log_graph=True)

    try:
        trainer.fit(model, datamodule=dm)
        return ObjectiveMetrics(
            psnr=trainer.callback_metrics["train/psnr"].item(),
            ms_ssim=trainer.callback_metrics["train/ms_ssim"].item(),
            lpips=trainer.callback_metrics["train/lpips"].item(),
        )
    except Exception as e:  # Out of memory
        if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
            return TrainingError.CUDA_OOM
        elif isinstance(e, ValueError):
            return TrainingError.FORWARD_RETURNED_NAN
        raise e


if __name__ == "__main__":
    import os

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

    ax_client = AxClient()
    ax_client.create_experiment(
        name="model_with_adam",
        parameters=model_params + adam_params,  # type: ignore
        objectives={
            "psnr": ObjectiveProperties(minimize=False, threshold=19.0),
            "ms_ssim": ObjectiveProperties(minimize=False, threshold=0.9),
            "lpips": ObjectiveProperties(minimize=True, threshold=0.1),
        },
    )

    for _ in range(10):
        parameters, trial_index = ax_client.get_next_trial()
        result = objective(parameters)

        if isinstance(result, TrainingError):
            metadata = {"errorName": result.name, "errorValue": result.value}
            print(f"metadata: {metadata}")
            ax_client.log_trial_failure(trial_index=trial_index, metadata=metadata)
        else:
            ax_client.complete_trial(trial_index=trial_index, raw_data=result.__dict__)

        print(f"Trial {trial_index}: {parameters}")

    pareto_frontier = ax_client.get_pareto_optimal_parameters()
    print(f"pareto_frontier: {pareto_frontier}")

    # DB_USER = os.environ["DB_USER"]
    # assert DB_USER is not None, "DB_USER environment variable must be set"
    # DB_PASS = os.environ["DB_PASS"]
    # assert DB_PASS is not None, "DB_PASS environment variable must be set"
    # DB_HOST = os.environ["DB_HOST"]
    # assert DB_HOST is not None, "DB_HOST environment variable must be set"
    # DB_PORT = os.environ["DB_PORT"]
    # assert DB_PORT is not None, "DB_PORT environment variable must be set"
    # DB_NAME = os.environ["DB_NAME"]
    # assert DB_NAME is not None, "DB_NAME environment variable must be set"
    # DB_URI = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # study = optuna.create_study(
    #     study_name=wandb_kwargs["group"],
    #     storage=RDBStorage(url=DB_URI, heartbeat_interval=60, grace_period=120),
    #     directions=[
    #         metric_direction for (_, metric_direction) in metric_names_and_directions
    #     ],
    #     pruner=SuccessiveHalvingPruner(),
    #     load_if_exists=True,
    # )
