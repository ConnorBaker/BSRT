from typing import Any, Dict, List, Optional

import bagua.torch_api as bagua
import torch
import torch.cuda
from ax import (
    MultiObjective,
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    Parameter,
)
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import build_stream_handler, get_logger
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from lightning_lite.utilities.seed import seed_everything

from bsrt.tuning.cli_parser import TunerConfig

from ..datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
from .cli_parser import CLI_PARSER, OptimizerName, SchedulerName, TunerConfig
from .lr_scheduler.cosine_annealing_warm_restarts import (
    COSINE_ANNEALING_WARM_RESTARTS_PARAMS,
)
from .lr_scheduler.exponential_lr import EXPONENTIAL_LR_PARAMS
from .lr_scheduler.reduce_lr_on_plateau import REDUCE_LR_ON_PLATEAU_PARAMS
from .model.bsrt import BSRT_PARAMS
from .objective import (
    LPIPS_DIVERGENCE_THRESHOLD,
    MS_SSIM_DIVERGENCE_THRESHOLD,
    PSNR_DIVERGENCE_THRESHOLD,
    TrainingError,
    objective,
)
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
    parameters: List[Parameter],
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

    # Model registry for creating multi-objective optimization models.
    from ax import Data, Experiment, Metric, Objective, ObjectiveThreshold, SearchSpace
    from ax.modelbridge.registry import Models
    from ax.runners.synthetic import SyntheticRunner
    from ax.service.utils.report_utils import exp_to_df

    metrics_and_thresholds = [
        (Metric(name="psnr", lower_is_better=False), PSNR_DIVERGENCE_THRESHOLD),
        (Metric(name="ms_ssim", lower_is_better=False), MS_SSIM_DIVERGENCE_THRESHOLD),
        (Metric(name="lpips", lower_is_better=True), LPIPS_DIVERGENCE_THRESHOLD),
    ]
    metric_names = [metric.name for metric, _ in metrics_and_thresholds]

    experiment = Experiment(
        name="pareto_experiment",
        search_space=SearchSpace(parameters),
        optimization_config=MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                [Objective(metric) for (metric, _) in metrics_and_thresholds],
            ),
            objective_thresholds=[
                ObjectiveThreshold(metric, threshold, relative=False)
                for (metric, threshold) in metrics_and_thresholds
            ],
        ),
        runner=SyntheticRunner(),
    )
    ax_client._set_experiment(experiment)

    N_INIT = 2 * (len(parameters) + 1)
    N_BATCH = 10
    BATCH_SIZE = 4
    sobol = Models.SOBOL(search_space=experiment.search_space)
    experiment.new_batch_trial(sobol.gen(N_INIT)).run()
    data = experiment.fetch_data()

    hv_list = []
    model = None
    for i in range(N_BATCH):
        model = Models.FULLYBAYESIANMOO(
            experiment=experiment,
            data=data,
            # use fewer num_samples and warmup_steps to speed up this tutorial
            num_samples=256,
            warmup_steps=512,
            torch_device=torch.device("cuda"),
            verbose=True,  # Set to True to print stats from MCMC
            disable_progbar=False,  # Set to False to print a progress bar from MCMC
        )
        generator_run = model.gen(BATCH_SIZE)
        trial = experiment.new_batch_trial(generator_run=generator_run, ttl_seconds=600)
        trial.run()
        data = Data.from_multiple_data([data, trial.fetch_data()])

        exp_df = exp_to_df(experiment)
        outcomes = torch.tensor(
            exp_df[metric_names].values, device=torch.device("cuda")
        )
        partitioning = DominatedPartitioning(
            ref_point=torch.tensor(
                [threshold for (_, threshold) in metrics_and_thresholds]
            ),
            Y=outcomes,
        )

        try:
            hv = partitioning.compute_hypervolume().item()
        except:
            hv = 0
            print("Failed to compute hv")
        hv_list.append(hv)
        print(f"Iteration: {i}, HV: {hv}")

    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    outcomes = df[metric_names].values
    print(f"Outcomes: {outcomes}")

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
    # We need to initialize the bagua backend before we can use the cached dataset
    torch.cuda.set_device(bagua.get_local_rank())
    if not bagua.communication.is_initialized():
        bagua.init_process_group()

    if config.precision == "bf16":
        precision = "bf16"
    elif config.precision == "16":
        precision = 16
    elif config.precision == "32":
        precision = 32
    else:
        raise ValueError(f"Unknown precision {config.precision}")

    datamodule = SyntheticZurichRaw2RgbDataModule(
        precision=precision,
        crop_size=256,
        data_dir="/home/connorbaker/ramdisk/datasets",
        burst_size=14,
        batch_size=config.batch_size,
        num_workers=-1,
        pin_memory=True,
        persistent_workers=True,
        cache_in_gb=40,
    )

    experiment_name = ax_client.experiment.name
    using_db = ax_client.db_settings_set
    for _ in range(config.num_trials):
        parameters, trial_index = ax_client.get_next_trial()

        try:
            result = objective(parameters, config, datamodule)

            if isinstance(result, TrainingError):
                metadata = {"errorName": result.name, "errorValue": result.value}
                logger.error(
                    f"Trial {trial_index} failed with error {result.name}: {result.value}"
                )
                ax_client.log_trial_failure(trial_index=trial_index, metadata=metadata)

                # If we got an unrecoverable error, we should stop the experiment.
                if result not in [
                    TrainingError.CUDA_OOM,
                    TrainingError.FORWARD_RETURNED_NAN,
                ]:
                    logger.error("Stopping experiment due to unrecoverable error")
                    return

            else:
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data=result.__dict__
                )

            if not using_db:
                ax_client.save_to_json_file(f"{experiment_name}.json")

        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed with error: {e}")
            ax_client.log_trial_failure(
                trial_index=trial_index, metadata={"error": str(e)}
            )
            raise e


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
