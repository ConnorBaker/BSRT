from typing import Any, Dict, List, Optional

import torch
import wandb
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import build_stream_handler, get_logger
from lightning_lite.utilities.seed import seed_everything

from .cli_parser import CLI_PARSER, OptimizerName, SchedulerName, TunerConfig
from .lr_scheduler.cosine_annealing_warm_restarts import (
    COSINE_ANNEALING_WARM_RESTARTS_PARAMS,
)
from .lr_scheduler.exponential_lr import EXPONENTIAL_LR_PARAMS
from .lr_scheduler.reduce_lr_on_plateau import REDUCE_LR_ON_PLATEAU_PARAMS
from .model.bsrt import BSRT_PARAMS
from .objective import TrainingError, objective
from .optimizer.decoupled_adamw import DECOUPLED_ADAMW_PARAMS
from .optimizer.decoupled_sgdw import DECOUPLED_SGDW_PARAMS

logger = get_logger("bsrt.tuning.tuner")
logger.addHandler(build_stream_handler())


def get_client(db_uri: Optional[str] = None, fallback_to_json: bool = True) -> AxClient:
    """
    Creates an AxClient object.

    Args:
        db_uri: The database URI. If this is None, the JSON fallback will be used.
        fallback_to_json: Whether to fallback to JSON if the database URI is None.

    Returns:
        An AxClient object.
    """
    if db_uri is not None:
        try:
            from ax.storage.sqa_store.structs import DBSettings

            ax_client = AxClient(
                torch_device=torch.device("cuda"),
                random_seed=0,
                db_settings=DBSettings(
                    url=db_uri,
                ),
            )
            logger.info("Connected to database")
            return ax_client
        except ModuleNotFoundError as e:
            logger.error(
                f"Failed to load AxClient from database due to missing dependencies: {e}. Falling back to local JSON storage."
            )
            if not fallback_to_json:
                raise e
        except Exception as e:
            logger.error(
                f"Failed to load AxClient from database due to error: {e}. Falling back to local JSON storage."
            )
            if not fallback_to_json:
                raise e

    ax_client = AxClient(
        torch_device=torch.device("cuda"),
        random_seed=0,
    )
    return ax_client


def create_experiment(
    ax_client: AxClient,
    experiment_name: str,
    parameters: List[Dict[str, Any]],
) -> None:
    """
    Creates an experiment in AxClient.

    Args:
        ax_client: The AxClient object.
        experiment_name: The name of the experiment.
        parameters: The parameters of the experiment.

    Returns:
        None
    """
    using_db = ax_client.db_settings_set
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
        name=experiment_name,
        parameters=parameters,
        objectives={
            "lpips": ObjectiveProperties(minimize=True),
            "psnr": ObjectiveProperties(minimize=False),
            "ms_ssim": ObjectiveProperties(minimize=False),
        },
    )
    if not using_db:
        ax_client.save_to_json_file(f"{experiment_name}.json")

    logger.info(
        f"Created and saved experiment `{experiment_name}` to {'database' if using_db else 'JSON file'}"
    )


# Returns a bool indicating whether the experiment was successfully loaded or whether a new one was created.
def load_experiment(
    ax_client: AxClient,
    experiment_name: str,
) -> bool:
    """
    Loads an experiment from AxClient.

    Args:
        ax_client: The AxClient object.
        experiment_name: The name of the experiment.

    Returns:
        A bool indicating whether the experiment was successfully loaded.
    """

    using_db = ax_client.db_settings_set
    try:
        if using_db:
            ax_client.load_experiment_from_database(experiment_name)
            logger.info(f"Loaded experiment `{experiment_name}` from database")
        else:
            ax_client.load_from_json_file(f"{experiment_name}.json")
            logger.info(f"Loaded experiment `{experiment_name}` from JSON file")
        return True
    except:
        return False


def trial_loops(ax_client: AxClient, config: TunerConfig) -> None:
    """
    Runs the trial loop.

    Args:
        ax_client: The AxClient object.
        num_trials: The number of trials to run.

    Returns:
        None
    """

    experiment_name = ax_client.experiment.name
    using_db = ax_client.db_settings_set
    for _ in range(config.num_trials):
        parameters, trial_index = ax_client.get_next_trial()

        try:
            result = objective(parameters, wandb_kwargs, config)

            if isinstance(result, TrainingError):
                metadata = {"errorName": result.name, "errorValue": result.value}
                logger.error(
                    f"Trial {trial_index} failed with error {result.name}: {result.value}"
                )
                ax_client.log_trial_failure(trial_index=trial_index, metadata=metadata)

                # If we got an unrecoverable error, we should stop the experiment.
                if (
                    result != TrainingError.CUDA_OOM
                    or result != TrainingError.FORWARD_RETURNED_NAN
                ):
                    logger.error("Stopping experiment due to unrecoverable error")
                    return

            else:
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data=result.__dict__
                )

            if not using_db:
                ax_client.save_to_json_file(f"{experiment_name}.json")

        except Exception as e:
            metadata = {
                "errorName": TrainingError.UNKNOWN_ERROR.name,
                "errorValue": str(e),
            }
            logger.error(
                f"Experiment {experiment_name} failed with error {TrainingError.UNKNOWN_ERROR.name}: {e}"
            )
            ax_client.log_trial_failure(trial_index=trial_index, metadata=metadata)
            return


if __name__ == "__main__":
    seed_everything(42)
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

    args = CLI_PARSER.parse_args()
    config = TunerConfig.from_args(args)

    wandb_kwargs = {
        "entity": "connorbaker",
        "project": "bsrt",
        # "group": Provided by the experiment name passed in from the command line
        "reinit": True,
        "settings": wandb.Settings(start_method="fork"),
    }
    wandb.login(key=config.wandb_api_key)
    wandb_kwargs["group"] = config.experiment_name

    using_db = True
    DB_URI = f"mysql+pymysql://{config.db_user}:{config.db_pass}@{config.db_host}:{config.db_port}/{config.db_name}"

    ax_client = get_client(DB_URI if using_db else None)

    model_params = BSRT_PARAMS

    # Find out which optimizer to use
    optimizer_name: OptimizerName = config.optimizer
    if optimizer_name == "DecoupledAdamW":
        optimizer_params = DECOUPLED_ADAMW_PARAMS
    elif optimizer_name == "DecoupledSGDW":
        optimizer_params = DECOUPLED_SGDW_PARAMS

    # Find out which scheduler to use
    scheduler_name: SchedulerName = config.scheduler
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler_params = COSINE_ANNEALING_WARM_RESTARTS_PARAMS
    elif scheduler_name == "ExponentialLR":
        scheduler_params = EXPONENTIAL_LR_PARAMS
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler_params = REDUCE_LR_ON_PLATEAU_PARAMS

    if not load_experiment(ax_client, config.experiment_name):
        create_experiment(
            ax_client=ax_client,
            experiment_name=config.experiment_name,
            parameters=model_params + optimizer_params + scheduler_params,
        )

    trial_loops(ax_client, config)

    if not using_db:
        ax_client.save_to_json_file(f"{config.experiment_name}.json")

    logger.info(
        f"Saved experiment `{config.experiment_name}` to {'database' if using_db else 'JSON file'}"
    )

    logger.info("Done!")
    logger.info(f"Best parameters: {ax_client.get_best_parameters()}")
