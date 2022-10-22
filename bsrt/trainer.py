from typing import Callable, List, Sequence, Tuple, cast, get_args

import optuna
import wandb
from datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
from model.bsrt import BSRT
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.storages import RDBStorage
from optuna.study import StudyDirection
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from typing_extensions import Literal

OptimizerName = Literal["AdamW", "SGD"]
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
wandbc = WeightsAndBiasesCallback(
    metric_name="final train loss",
    wandb_kwargs=wandb_kwargs,
    as_multirun=True,
)


def objective(cli: LightningCLI, trial: optuna.trial.Trial) -> Sequence[float]:
    # Update the existing configurations from the CLI
    # NOTE: When updating values which do not require initialization, use
    # cli.config. For classes like the logger or callbacks, use cli.
    # config_init because we are passing in a new, initialized instances of
    # the class, not the class path and arguments.
    hyperparameters = {}

    ### Model Hyperparameters ###
    attn_drop_rate = trial.suggest_float("attn_drop_rate", 0.0, 1.0)
    cli.config.model.attn_drop_rate = attn_drop_rate
    hyperparameters["attn_drop_rate"] = attn_drop_rate

    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0)
    cli.config.model.drop_path_rate = drop_path_rate
    hyperparameters["drop_path_rate"] = drop_path_rate

    drop_rate = trial.suggest_float("drop_rate", 0.0, 1.0)
    cli.config.model.drop_rate = drop_rate
    hyperparameters["drop_rate"] = drop_rate

    mlp_ratio_pow = trial.suggest_int("mlp_ratio_pow", 1, 5)
    hyperparameters["mlp_ratio_pow"] = mlp_ratio_pow
    mlp_ratio = 2.0**mlp_ratio_pow
    cli.config.model.mlp_ratio = mlp_ratio
    hyperparameters["mlp_ratio"] = mlp_ratio

    # Must be a multiple of flow alignment groups
    num_features_pow = trial.suggest_int("num_features_pow", 5, 10)
    hyperparameters["num_features_pow"] = num_features_pow
    num_features = 2**num_features_pow
    cli.config.model.num_features = num_features
    hyperparameters["num_features"] = num_features

    use_qk_scale = trial.suggest_categorical("use_qk_scale", [True, False])
    hyperparameters["use_qk_scale"] = use_qk_scale
    if use_qk_scale:
        qk_scale = trial.suggest_float("qk_scale", 1e-8, 1.0)
        cli.config.model.qk_scale = qk_scale
        hyperparameters["qk_scale"] = qk_scale
    else:
        cli.config.model.qk_scale = None

    qkv_bias = trial.suggest_categorical("qkv_bias", [True, False])
    cli.config.model.qkv_bias = qkv_bias
    hyperparameters["qkv_bias"] = qkv_bias

    ### Loss Hyperparameters ###
    # gan_k = trial.suggest_int("gan_k", 1, 5)
    # hyperparameters["gan_k"] = gan_k

    # gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    # hyperparameters["gamma"] = gamma

    ### Optimizer Hyperparameters ###
    # decay_milestones: Categorical = tune.choice(
    #     [[x, y] for x in range(40, 300, 20) for y in range(80, 400, 20) if x < y]
    # )
    optimizer_name: OptimizerName = cast(
        OptimizerName,
        trial.suggest_categorical("optimizer_name", get_args(OptimizerName)),
    )
    hyperparameters["optimizer_name"] = optimizer_name
    match optimizer_name:
        case "AdamW":
            adam_lr = trial.suggest_float("adam_lr", 1e-5, 1e-1, log=True)
            cli.config.model.lr = adam_lr
            hyperparameters["adam_lr"] = adam_lr

            adam_beta_gradient = trial.suggest_float(
                "adam_beta_gradient", 1e-4, 1.0, log=True
            )
            hyperparameters["adam_beta_gradient"] = adam_beta_gradient

            adam_beta_square = trial.suggest_float(
                "adam_beta_square", adam_beta_gradient, 1.0, log=True
            )
            hyperparameters["adam_beta_square"] = adam_beta_square

            adam_epsilon = trial.suggest_float("adam_epsilon", 1e-10, 1e-3, log=True)
            hyperparameters["adam_epsilon"] = adam_epsilon

            adam_weight_decay = trial.suggest_float(
                "adam_weight_decay", 1e-10, 1e-4, log=True
            )
            hyperparameters["adam_weight_decay"] = adam_weight_decay

            adam_amsgrad = bool(
                trial.suggest_categorical("adam_amsgrad", [True, False])
            )
            hyperparameters["adam_amsgrad"] = adam_amsgrad

            optimizer = AdamW(
                cli.config_init.model.parameters(),
                lr=adam_lr,
                betas=(adam_beta_gradient, adam_beta_square),
                eps=adam_epsilon,
                weight_decay=adam_weight_decay,
                amsgrad=adam_amsgrad,
            )
        case "SGD":
            sgd_lr = trial.suggest_float("sgd_lr", 1e-5, 1e-1, log=True)
            cli.config.model.lr = sgd_lr
            hyperparameters["sgd_lr"] = sgd_lr

            sgd_momentum = trial.suggest_float("sgd_momentum", 1e-4, 1.0, log=True)
            hyperparameters["sgd_momentum"] = sgd_momentum

            sgd_dampening = trial.suggest_float("sgd_dampening", 1e-4, 1.0, log=True)
            hyperparameters["sgd_dampening"] = sgd_dampening

            sgd_weight_decay = trial.suggest_float(
                "sgd_weight_decay", 1e-10, 1e-4, log=True
            )
            hyperparameters["sgd_weight_decay"] = sgd_weight_decay

            optimizer = SGD(
                cli.config_init.model.parameters(),
                lr=sgd_lr,
                momentum=sgd_momentum,
                dampening=sgd_dampening,
                weight_decay=sgd_weight_decay,
            )

    cli.config_init.optimizer = optimizer

    ### Scheduler Hyperparameters ###
    scheduler_name: SchedulerName = cast(
        SchedulerName,
        trial.suggest_categorical("scheduler_name", get_args(SchedulerName)),
    )
    hyperparameters["scheduler_name"] = scheduler_name
    match scheduler_name:
        case "ExponentialLR":
            gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
            hyperparameters["gamma"] = gamma

            lr_scheduler = ExponentialLR(
                optimizer,
                gamma=gamma,
            )

        case "ReduceLROnPlateau":
            factor = trial.suggest_float("factor", 1e-4, 1.0, log=True)
            hyperparameters["factor"] = factor

            lr_scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=10)

        case "OneCycleLR":
            match optimizer_name:
                case "AdamW":
                    max_lr = trial.suggest_float("max_lr", adam_lr, 1e-1, log=True)
                case "SGD":
                    max_lr = trial.suggest_float("max_lr", sgd_lr, 1e-1, log=True)

            hyperparameters["max_lr"] = max_lr

            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=cli.config.data.batch_size,
                epochs=cli.config.trainer.max_epochs,
            )

        case "CosineAnnealingWarmRestarts":
            T_0 = trial.suggest_int("T_0", 1, 1000)
            hyperparameters["T_0"] = T_0

            T_mult = trial.suggest_int("T_mult", 1, 10)
            hyperparameters["T_mult"] = T_mult

            eta_min = trial.suggest_float("eta_min", 1e-10, 1e-3, log=True)
            hyperparameters["eta_min"] = eta_min

            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
            )

    cli.config_init.lr_scheduler = lr_scheduler

    swa_lrs = trial.suggest_float("swa_lr", 1e-5, 1e-1, log=True)
    hyperparameters["swa_lr"] = swa_lrs
    swa = StochasticWeightAveraging(swa_lrs=swa_lrs)

    ### Setup the trainer ###
    if cli.config_init.trainer.callbacks is None:
        cli.config_init.trainer.callbacks = []

    cli.config_init.trainer.callbacks.append(swa)
    cli.config_init.trainer.callbacks.extend(
        PyTorchLightningPruningCallback(trial, monitor=metric_name)
        for (metric_name, _) in metric_names_and_directions
    )

    cli.instantiate_classes()
    model = cli.model
    trainer = cli.trainer
    logger = trainer.logger = WandbLogger(**wandb_kwargs)
    assert isinstance(logger, WandbLogger), "Logger should be set to the WandbLogger"

    logger.log_hyperparams(hyperparameters)
    logger.watch(model, log="all", log_graph=True)
    trainer.fit(model, datamodule=cli.datamodule)

    return [
        trainer.callback_metrics[metric_name].item()
        for (metric_name, _) in metric_names_and_directions
    ]


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

    cli = LightningCLI(
        BSRT,
        SyntheticZurichRaw2RgbDataModule,
        run=False,
        save_config_callback=None,
    )

    metric_names_and_directions: List[Tuple[str, StudyDirection]] = [
        ("train/lpips", StudyDirection.MINIMIZE),
        ("train/psnr", StudyDirection.MAXIMIZE),
        ("train/ms_ssim", StudyDirection.MAXIMIZE),
    ]

    DB_USER = os.environ["DB_USER"]
    assert DB_USER is not None, "DB_USER environment variable must be set"
    DB_PASS = os.environ["DB_PASS"]
    assert DB_PASS is not None, "DB_PASS environment variable must be set"
    DB_HOST = os.environ["DB_HOST"]
    assert DB_HOST is not None, "DB_HOST environment variable must be set"
    DB_PORT = os.environ["DB_PORT"]
    assert DB_PORT is not None, "DB_PORT environment variable must be set"
    DB_NAME = os.environ["DB_NAME"]
    assert DB_NAME is not None, "DB_NAME environment variable must be set"
    DB_URI = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    study = optuna.create_study(
        study_name=wandb_kwargs["group"],
        storage=RDBStorage(url=DB_URI, heartbeat_interval=60, grace_period=120),
        directions=[
            metric_direction for (_, metric_direction) in metric_names_and_directions
        ],
        pruner=SuccessiveHalvingPruner(),
        load_if_exists=True,
    )

    # wandbc.track_in_wandb is a decorator which adds trial/run number to the name of the run
    wrapper: Callable[[optuna.Trial], Sequence[float]] = wandbc.track_in_wandb()(
        lambda trial: objective(cli, trial)
    )

    while True:
        try:
            study.optimize(
                wrapper,
                n_trials=1000,
                # Catch CUDA OOM errors
                catch=(RuntimeError,),
                callbacks=[wandbc],
                n_jobs=1,
                show_progress_bar=True,
            )
        except TypeError:
            # The WandbCallback expects there to be at least one completed trial.
            # If there are no completed trials, it performs a len() on an NoneType (because there are no trials).
            continue
