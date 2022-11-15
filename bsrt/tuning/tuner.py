from pathlib import Path

import torch
import torch.cuda
import wandb
from lightning_lite.utilities.seed import seed_everything
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA

from bsrt.tuning.cli_parser import CLI_PARSER, TunerConfig
from bsrt.tuning.lr_scheduler.one_cycle_lr import OneCycleLRConfigSpace
from bsrt.tuning.model.bsrt import BSRTConfigSpace
from bsrt.tuning.optimizer.adamw import AdamWConfigSpace

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
    tuner_config = TunerConfig.from_args(args)

    wandb.setup()
    wandb.login(key=tuner_config.wandb_api_key)

    bsrt_config_space = BSRTConfigSpace()
    adamw_config_space = AdamWConfigSpace()
    # TODO: Incorrect calculation of the number of steps per epoch
    one_cycle_lr_config_space = OneCycleLRConfigSpace(
        tuner_config.max_epochs,
        int((tuner_config.limit_train_batches * 46839 * 0.8) / tuner_config.batch_size),
    )
    config_space = (
        tuner_config.__dict__
        | bsrt_config_space.to_dict()
        | adamw_config_space.to_dict()
        | one_cycle_lr_config_space.to_dict()
    )

    scheduler = MOASHA(
        time_attr="epoch",
        max_t=tuner_config.max_epochs,
        config_space=config_space,
        metrics=["val/lpips", "val/ms_ssim", "val/psnr"],
        mode=["min", "max", "max"],
    )

    # tuner_dir = Path(__file__).parent / "syne_tune"
    # tuner_dir.mkdir(exist_ok=True)

    entry_point = Path(__file__).parent / "objective.py"
    trial_backend = LocalBackend(entry_point=entry_point.as_posix())
    # trial_backend.set_path(tuner_dir.as_posix())

    stop_criterion = StoppingCriterion()
    tuner = Tuner(
        tuner_name="blarg",
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
        max_failures=1000,
        sleep_time=5.0,
        save_tuner=True,
        suffix_tuner_name=False,
    )
    # if (tuner_dir / "tuner.dill").exists():
    #     tuner.load(tuner_dir.as_posix())
    tuner.run()
