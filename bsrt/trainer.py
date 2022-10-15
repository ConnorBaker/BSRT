from typing import cast, get_args

import optuna
from datasets.synthetic_train_zurich_raw2rgb_data_module import (
    SyntheticTrainZurichRaw2RgbDatasetDataModule,
)
from model.bsrt import BSRT
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.study import StudyDirection
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim import SGD, Adam, AdamW, RMSprop
from typing_extensions import Literal

OptimizerName = Literal["Adam", "AdamW", "RMSprop", "SGD"]

if __name__ == "__main__":
    import os

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["TORCH_CUDNN_V8_API_DEBUG"] = "1"

    import torch.backends.cuda
    import torch.backends.cudnn

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    wandb_kwargs = {
        "entity": "connorbaker",
        "project": "bsrt",
        "group": "PL",
        "dir": None,
        "reinit": True,
    }
    wandbc = WeightsAndBiasesCallback(
        metric_name="final validation accuracy",
        wandb_kwargs=wandb_kwargs,
        as_multirun=True,
    )

    cli = LightningCLI(
        BSRT,
        SyntheticTrainZurichRaw2RgbDatasetDataModule,
        run=False,
        save_config_callback=None,
    )

    # Decorator adds trial/run number to the name of the run
    @wandbc.track_in_wandb()
    def objective(trial: optuna.trial.Trial) -> float:
        # Update the existing configurations from the CLI
        # NOTE: When updating values which do not require initialization, use
        # cli.config. For classes like the logger or callbacks, use cli.
        # config_init because we are passing in a new, initialized instances of
        # the class, not the class path and arguments.
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        cli.config.model.lr = lr

        optimizer_name: OptimizerName = cast(
            OptimizerName,
            trial.suggest_categorical("optimizer_name", get_args(OptimizerName)),
        )
        match optimizer_name:
            case "Adam":
                optimizer = Adam(cli.config_init.model.parameters(), lr=lr)
            case "AdamW":
                optimizer = AdamW(cli.config_init.model.parameters(), lr=lr)
            case "RMSprop":
                optimizer = RMSprop(cli.config_init.model.parameters(), lr=lr)
            case "SGD":
                optimizer = SGD(cli.config_init.model.parameters(), lr=lr)

        cli.config_init.optimizer = optimizer

        drop_rate = trial.suggest_float("drop_rate", 1e-8, 1.0, log=True)
        cli.config.model.drop_rate = drop_rate

        if cli.config_init.trainer.callbacks is None:
            cli.config_init.trainer.callbacks = []
        cli.config_init.trainer.callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="train/loss")
        )

        cli.instantiate_classes()
        model = cli.model
        trainer = cli.trainer
        logger = trainer.logger = WandbLogger(**wandb_kwargs)
        assert isinstance(
            logger, WandbLogger
        ), "Logger should be set to the WandbLogger"

        logger.log_hyperparams(dict(optimizer=optimizer, lr=lr, drop_rate=drop_rate))
        trainer.fit(model, datamodule=cli.datamodule)

        return trainer.callback_metrics["train/loss"].item()

    study = optuna.create_study(
        study_name="bsrt",
        storage="sqlite:///bsrt.db",
        direction=StudyDirection.MINIMIZE,
        pruner=MedianPruner(),
        load_if_exists=True,
    )

    if study.trials != []:
        print("Found existing study, resuming...")
        print(f"Study name: {study.study_name}")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best trial params: {study.best_trial.params}")

    study.optimize(
        objective,
        n_trials=100,
        callbacks=[wandbc],
        n_jobs=1,
        show_progress_bar=True,
    )

    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best trial params: {study.best_trial.params}")

    # TODO: Add a way to save the best model and the figures from the best run
    # for param_name, param_value in study.best_trial.params.items():
    #     wandb.run.summary[f"best_{param_name}"] = param_value

    # wandb.run.summary["best accuracy"] = study.best_trial.value
    # wandb.log(
    #     {
    #         "optuna_optimization_history": optuna.visualization.plot_optimization_history(
    #             study
    #         ),
    #         "optuna_param_importances": optuna.visualization.plot_param_importances(
    #             study
    #         ),
    #     }
    # )
    # wandb.finish()
