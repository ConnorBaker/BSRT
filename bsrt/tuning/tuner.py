from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union

import torch
import wandb
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.bagua import BaguaStrategy
from pytorch_lightning.trainer import Trainer
from typing_extensions import Literal

from ..datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
from ..lighting_bsrt import LightningBSRT
from .model.bsrt import BSRT_PARAMS, BSRTParams
from .optimizer.adam import ADAM_PARAM_CONSTRAINTS, ADAM_PARAMS, AdamParams
from .optimizer.sgd import SGDParams
from .utilities import filter_and_remove_from_keys

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
    seed_everything(42)

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
        bsrt_params = BSRTParams(**filter_and_remove_from_keys("bsrt_params", params))
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

    wandb_logger = WandbLogger(**wandb_kwargs)
    wandb_logger.log_hyperparams(params)
    wandb_logger.watch(model, log="all", log_graph=True)

    datamodule = SyntheticZurichRaw2RgbDataModule(
        precision=32,
        crop_size=256,
        data_dir="/home/connorbaker/ramdisk/datasets",
        burst_size=14,
        batch_size=32,
        num_workers=-1,
        pin_memory=True,
        persistent_workers=True,
        cache_in_gb=40,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=32,
        enable_checkpointing=False,
        strategy=BaguaStrategy(algorithm="gradient_allreduce"),
        limit_train_batches=10,
        limit_val_batches=1,
        max_epochs=5,
        detect_anomaly=False,
        enable_model_summary=True,
        enable_progress_bar=False,
        logger=wandb_logger,
        replace_sampler_ddp=False,
    )

    try:
        trainer.fit(model=model, datamodule=datamodule)
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

    from ax.utils.common.logger import build_stream_handler, get_logger

    logger = get_logger("bsrt.tuning.tuner")
    logger.addHandler(build_stream_handler())

    EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME")
    assert EXPERIMENT_NAME is not None, "EXPERIMENT_NAME must be set"
    using_db = True

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    assert WANDB_API_KEY is not None, "WANDB_API_KEY must be set"
    wandb.login(key=WANDB_API_KEY)

    try:
        from ax.storage.sqa_store.structs import DBSettings

        DB_USER = os.environ.get("DB_USER")
        assert DB_USER is not None, "DB_USER environment variable must be set"
        DB_PASS = os.environ.get("DB_PASS")
        assert DB_PASS is not None, "DB_PASS environment variable must be set"
        DB_HOST = os.environ.get("DB_HOST")
        assert DB_HOST is not None, "DB_HOST environment variable must be set"
        DB_PORT = os.environ.get("DB_PORT")
        assert DB_PORT is not None, "DB_PORT environment variable must be set"
        DB_NAME = os.environ.get("DB_NAME")
        assert DB_NAME is not None, "DB_NAME environment variable must be set"
        DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        ax_client = AxClient(
            torch_device=torch.device("cuda"),
            random_seed=0,
            db_settings=DBSettings(
                url=DB_URI,
            ),
        )
        logger.info("Connected to database")
    except ModuleNotFoundError as e:
        logger.error(
            f"Failed to load experiment `{EXPERIMENT_NAME}` from database due to missing dependencies: {e}. Falling back to local JSON storage."
        )
        ax_client = AxClient(
            torch_device=torch.device("cuda"),
            random_seed=0,
        )
        using_db = False

    except AssertionError as e:
        logger.error(
            f"Failed to load experiment `{EXPERIMENT_NAME}` from database due to missing environment variable: {e}. Falling back to local JSON storage."
        )
        ax_client = AxClient(
            torch_device=torch.device("cuda"),
            random_seed=0,
        )
        using_db = False

    try:
        if using_db:
            ax_client.load_experiment_from_database(EXPERIMENT_NAME)
            logger.info(f"Loaded experiment `{EXPERIMENT_NAME}` from database")
        else:
            ax_client.load_from_json_file(f"{EXPERIMENT_NAME}.json")
            logger.info(f"Loaded experiment `{EXPERIMENT_NAME}` from JSON file")

    except:
        logger.error(
            f"Failed to load experiment `{EXPERIMENT_NAME}` from {'database' if using_db else 'JSON file'}. Creating new experiment."
        )
        if using_db:
            from ax.storage.sqa_store.db import (
                create_all_tables,
                get_engine,
                init_engine_and_session_factory,
            )

            init_engine_and_session_factory(url=ax_client.db_settings.url)
            engine = get_engine()
            create_all_tables(engine)
            logger.info("Created database tables")

        ax_client.create_experiment(
            name=EXPERIMENT_NAME,
            parameters=BSRT_PARAMS + ADAM_PARAMS,
            parameter_constraints=ADAM_PARAM_CONSTRAINTS,
            objectives={
                "lpips": ObjectiveProperties(minimize=True),
            },
            tracking_metric_names=["psnr", "ms_ssim"],
            outcome_constraints=[
                "psnr >= 20.0",
                "ms_ssim >= 0.95",
            ],
        )
        if not using_db:
            ax_client.save_to_json_file(f"{EXPERIMENT_NAME}.json")

        logger.info(
            f"Created and saved experiment `{EXPERIMENT_NAME}` to {'database' if using_db else 'JSON file'}"
        )

    for _ in range(1):
        parameters, trial_index = ax_client.get_next_trial()
        result = objective(parameters)

        if isinstance(result, TrainingError):
            metadata = {"errorName": result.name, "errorValue": result.value}
            logger.error(
                f"Trial {trial_index} failed with error {result.name}: {result.value}"
            )
            ax_client.log_trial_failure(trial_index=trial_index, metadata=metadata)

        else:
            ax_client.complete_trial(trial_index=trial_index, raw_data=result.__dict__)

        if not using_db:
            ax_client.save_to_json_file(f"{EXPERIMENT_NAME}.json")
