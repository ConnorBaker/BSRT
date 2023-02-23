# Example usage:
# python -m bsrt.tuning.tuner --experiment_name bsrt-exp --optimizer AdamW --scheduler OneCycleLR --precision float32 --data_dir "/home/connorbaker/datasets" --limit_train_batches 1.0 --limit_val_batches 1.0 --batch_size 8 --max_epochs 100 --wandb_api_key "YOUR_API_KEY"

from pathlib import Path

from lightning_lite.utilities.seed import seed_everything
from mfsr_utils.datasets.zurich_raw2rgb import ZurichRaw2Rgb
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.util import experiment_path

import wandb
from bsrt.tuning.cli_parser import CLI_PARSER, OptimizerName, SchedulerName, TunerConfig
from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsConfigSpace,
)
from bsrt.tuning.lr_scheduler.exponential_lr import ExponentialLRConfigSpace
from bsrt.tuning.lr_scheduler.one_cycle_lr import OneCycleLRConfigSpace
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauConfigSpace
from bsrt.tuning.model.bsrt import BSRTConfigSpace
from bsrt.tuning.optimizer.adamw import AdamWConfigSpace
from bsrt.tuning.optimizer.sgd import SGDConfigSpace


def get_optimizer_config_space(optimizer_name: OptimizerName) -> ConfigSpace:
    if optimizer_name == "AdamW":
        return AdamWConfigSpace()
    elif optimizer_name == "SGD":
        return SGDConfigSpace()
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


def get_scheduler_config_space(scheduler_name: SchedulerName) -> ConfigSpace:
    if scheduler_name == "OneCycleLR":
        return OneCycleLRConfigSpace()
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestartsConfigSpace()
    elif scheduler_name == "ExponentialLR":
        return ExponentialLRConfigSpace()
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateauConfigSpace()
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")


if __name__ == "__main__":
    seed_everything(42)

    args = CLI_PARSER.parse_args()
    tuner_config = TunerConfig.from_args(args)
    ZurichRaw2Rgb.download_to(Path(tuner_config.data_dir))

    wandb.setup()
    wandb.login(key=tuner_config.wandb_api_key)

    bsrt_config_space = BSRTConfigSpace()
    optimizer_config_space = get_optimizer_config_space(tuner_config.optimizer)
    scheduler_config_space = get_scheduler_config_space(tuner_config.scheduler)
    config_space = (
        tuner_config.__dict__
        | bsrt_config_space.to_dict()
        | optimizer_config_space.to_dict()
        | scheduler_config_space.to_dict()
    )
    scheduler = MOASHA(
        time_attr="epoch",
        max_t=tuner_config.max_epochs,
        config_space=config_space,
        metrics=["val/lpips", "val/ms_ssim", "val/psnr"],
        mode=["min", "max", "max"],
    )

    entry_point = Path(__file__).parent / "objective.py"
    trial_backend = LocalBackend(entry_point=entry_point.as_posix())

    tuner_name = tuner_config.experiment_name or "-".join(
        ["BSRT", tuner_config.optimizer, tuner_config.scheduler, scheduler.__class__.__name__]
    )
    tuner_path = Path(experiment_path(tuner_name=tuner_name))
    try:
        tuner = Tuner.load(tuner_path.as_posix())
    except FileNotFoundError:
        tuner = Tuner(
            tuner_name=tuner_name,
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(),
            n_workers=1,
            max_failures=100,
            sleep_time=5.0,
            save_tuner=True,
            suffix_tuner_name=False,
        )

    tuner.run()
