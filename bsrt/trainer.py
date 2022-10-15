from typing import cast, get_args

import optuna
from datasets.synthetic_train_zurich_raw2rgb_data_module import \
    SyntheticTrainZurichRaw2RgbDatasetDataModule
from model.bsrt import BSRT
from optuna.integration.pytorch_lightning import \
    PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.pruners import (HyperbandPruner, MedianPruner,
                            SuccessiveHalvingPruner)
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
        "group": "32-hyperparameter-tuning",
        "dir": None,
        "reinit": True,
    }
    wandbc = WeightsAndBiasesCallback(
        metric_name="final train loss",
        wandb_kwargs=wandb_kwargs,
        as_multirun=True,
    )

    cli = LightningCLI(BSRT, SyntheticTrainZurichRaw2RgbDatasetDataModule, run=False, save_config_callback=None)

    # Decorator adds trial/run number to the name of the run
    @wandbc.track_in_wandb()
    def objective(trial: optuna.trial.Trial) -> float:
        # Update the existing configurations from the CLI
        # NOTE: When updating values which do not require initialization, use
        # cli.config. For classes like the logger or callbacks, use cli.
        # config_init because we are passing in a new, initialized instances of
        # the class, not the class path and arguments.
        hyperparameters = {}

        ### Model Hyperparameters ###
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        cli.config.model.lr = lr
        hyperparameters["lr"] = lr

        attn_drop_rate = trial.suggest_float("attn_drop_rate", 0.0, 1.0)
        cli.config.model.attn_drop_rate = attn_drop_rate
        hyperparameters["attn_drop_rate"] = attn_drop_rate

        drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0)
        cli.config.model.drop_path_rate = drop_path_rate
        hyperparameters["drop_path_rate"] = drop_path_rate

        drop_rate = trial.suggest_float("drop_rate", 1e-8, 1.0, log=True)
        cli.config.model.drop_rate = drop_rate
        hyperparameters["drop_rate"] = drop_rate

        mlp_ratio = trial.suggest_float("mlp_ratio", 1.0, 32.0, log=True)
        cli.config.model.mlp_ratio = mlp_ratio
        hyperparameters["mlp_ratio"] = mlp_ratio
        
        # Must be a multiple of flow alignment groups
        num_features_pow = trial.suggest_int("num_features_pow", 5, 10)        
        hyperparameters["num_features_pow"] = num_features_pow
        num_features = 2 ** num_features_pow
        cli.config.model.num_features = num_features
        hyperparameters["num_features"] = num_features

        use_qk_scale = trial.suggest_categorical("use_qk_scale", [True, False])
        hyperparameters["use_qk_scale"] = use_qk_scale
        if use_qk_scale:
            qk_scale = trial.suggest_float("qk_scale", 1e-8, 1.0, log=True)
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
        momentum = trial.suggest_float("momentum", 1e-4, 1.0, log=True)
        hyperparameters["momentum"] = momentum

        beta_gradient = trial.suggest_float("beta_gradient", 1e-4, 1.0, log=True)
        hyperparameters["beta_gradient"] = beta_gradient

        beta_square = trial.suggest_float("beta_square", 1e-4, 1.0, log=True)
        hyperparameters["beta_square"] = beta_square

        epsilon = trial.suggest_float("epsilon", 1e-10, 1e-4, log=True)
        hyperparameters["epsilon"] = epsilon

        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-4, log=True)
        hyperparameters["weight_decay"] = weight_decay
        
        optimizer_name: OptimizerName = cast(
            OptimizerName,
            trial.suggest_categorical("optimizer_name", get_args(OptimizerName)),
        )
        hyperparameters["optimizer_name"] = optimizer_name
        match optimizer_name:
            case "Adam":
                optimizer = Adam(cli.config_init.model.parameters(), lr=lr, betas=(beta_gradient, beta_square), eps=epsilon, weight_decay=weight_decay)
            case "AdamW":
                optimizer = AdamW(cli.config_init.model.parameters(), lr=lr, betas=(beta_gradient, beta_square), eps=epsilon, weight_decay=weight_decay)
            case "RMSprop":
                alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
                hyperparameters["alpha"] = alpha
                optimizer = RMSprop(cli.config_init.model.parameters(), lr=lr, alpha=alpha, eps=epsilon, weight_decay=weight_decay, momentum=momentum)
            case "SGD":
                optimizer = SGD(cli.config_init.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        cli.config_init.optimizer = optimizer


        ### Setup the trainer ###
        if cli.config_init.trainer.callbacks is None:
            cli.config_init.trainer.callbacks = []
        cli.config_init.trainer.callbacks.append(PyTorchLightningPruningCallback(trial, monitor="train/loss"))
        

        cli.instantiate_classes()
        model = cli.model
        trainer = cli.trainer
        logger = trainer.logger = WandbLogger(**wandb_kwargs)
        assert isinstance(
            logger, WandbLogger
        ), "Logger should be set to the WandbLogger"

        logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=cli.datamodule)

        return trainer.callback_metrics["train/loss"].item()

    study = optuna.create_study(
        study_name="bsrt-32-hyperparameter-tuning",
        storage="sqlite:///bsrt.db",
        direction=StudyDirection.MINIMIZE,
        pruner=MedianPruner(),
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=1000,
        callbacks=[wandbc],
        n_jobs=1,
    )

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