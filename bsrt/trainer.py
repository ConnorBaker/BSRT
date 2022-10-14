import os

import optuna
import torch
import torch.backends.cuda
import torch.backends.cudnn
from datasets.synthetic_train_zurich_raw2rgb_data_module import \
    SyntheticTrainZurichRaw2RgbDatasetDataModule
from model.bsrt import BSRT
from optuna.integration.pytorch_lightning import \
    PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
os.environ["NCCL_SOCKET_NTHREADS"] = "4"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TORCH_CUDNN_V8_API_DEBUG"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True



if __name__ == "__main__":
    # Initialise wandb callback
    wandb_kwargs = {
        "project": "bsrt",
        "entity": "connorbaker",
        "reinit": True,
    }

    wandbc = WeightsAndBiasesCallback(
        metric_name="final validation accuracy", wandb_kwargs=wandb_kwargs, as_multirun=True
    )

    cli = LightningCLI(BSRT, SyntheticTrainZurichRaw2RgbDatasetDataModule, run=False)

    # Add decorator to track objective
    @wandbc.track_in_wandb()
    def objective(trial: optuna.trial.Trial) -> float:
        # Update the existing configurations from the CLI
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
        cli.config.optimizer = optimizer
        
        lr = trial.suggest_float("lr", 1e-8, 1.0, log=True)
        cli.config.model.lr = lr
        
        drop_rate = trial.suggest_float("drop_rate", 1e-8, 1.0, log=True)
        cli.config.model.drop_rate = drop_rate

        # TODO: Does this need to be defined inisde the objective function?
        wandb_logger = WandbLogger(project="MNIST", log_model="all")

        
        # TODO: lightning_lite.utilities.exceptions.MisconfigurationException: accelerator set through both strategy class and accelerator flag, choose one
        cli.instantiate_classes()
        model = cli.model
        trainer = cli.instantiate_trainer(logger=wandb_logger, limit_val_batches=10, max_epochs=100,callbacks=cli.config.trainer.callbacks + [PyTorchLightningPruningCallback(trial, monitor="loss")])

        hyperparameters = dict(optimizer=optimizer, lr=lr, drop_rate=drop_rate)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=cli.datamodule)

        return trainer.callback_metrics["loss"].item()
    
    
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=5, timeout=600, callbacks=[wandbc], n_jobs=1, show_progress_bar=True)
