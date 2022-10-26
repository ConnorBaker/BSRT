from functools import partial
from logging import StreamHandler
from sys import stdout

import bagua.torch_api as bagua
import optuna
import torch
import torch.cuda
import wandb
from optuna.logging import get_logger
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from pytorch_lightning.utilities.seed import seed_everything

from ..datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
from .cli_parser import CLI_PARSER, TunerConfig
from .objective import objective

# Add stream handler of stdout to show the messages
logger = get_logger("optuna").addHandler(StreamHandler(stdout))

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
    wandb.login(key=config.wandb_api_key)

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

    study_name = config.experiment_name
    study = optuna.create_study(
        storage=RDBStorage(
            url=config.db_uri,
            heartbeat_interval=60,
            grace_period=180,
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        ),
        sampler=TPESampler(
            seed=42,
            multivariate=True,
            group=True,
            n_startup_trials=10,
            n_ei_candidates=24,
        ),
        pruner=HyperbandPruner(
            min_resource=2,
            max_resource=config.max_epochs,
            reduction_factor=3,
            bootstrap_count=2,
        ),
        study_name=study_name,
        load_if_exists=True,
        directions=["maximize", "maximize", "minimize"],
    )

    study.optimize(
        partial(objective, config, datamodule),  # type: ignore
        n_trials=config.num_trials,
    )
