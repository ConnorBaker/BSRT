from functools import partial
from logging import StreamHandler
from sys import stdout

import optuna
import torch
import torch.cuda
import wandb
from lightning_lite.utilities.seed import seed_everything
from optuna.logging import get_logger
from optuna.storages import RDBStorage, RetryFailedTrialCallback

from bsrt.datasets.synthetic_zurich_raw2rgb import SyntheticZurichRaw2Rgb
from bsrt.tuning.cli_parser import CLI_PARSER, PRECISION_MAP, TunerConfig
from bsrt.tuning.objective import objective

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

    datamodule = SyntheticZurichRaw2Rgb(
        precision=PRECISION_MAP[config.precision],
        crop_size=256,
        data_dir=config.data_dir,
        burst_size=14,
        batch_size=config.batch_size,
        num_workers=-1,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    study_name = config.experiment_name
    study = optuna.create_study(
        storage=RDBStorage(
            url=config.db_uri,
            heartbeat_interval=60,
            grace_period=180,
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        )
        if config.db_uri != ""
        else None,
        study_name=study_name,
        load_if_exists=True,
        directions=["maximize", "maximize", "minimize"],
    )

    wandb.setup()
    wandb.login(key=config.wandb_api_key)

    study.optimize(
        partial(objective, config, datamodule),  # type: ignore
        n_trials=config.num_trials,
    )
