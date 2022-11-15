from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from typing_extensions import Literal, get_args

OptimizerName = Literal["AdamW", "SGD"]
SchedulerName = Literal[
    "CosineAnnealingWarmRestarts", "ExponentialLR", "OneCycleLR", "ReduceLROnPlateau"
]
PrecisionName = Literal["bfloat16", "float16", "float32", "float64"]
PRECISION_MAP: Mapping[PrecisionName, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass
class TunerConfig:
    wandb_api_key: str

    experiment_name: Optional[str]
    optimizer: OptimizerName
    scheduler: SchedulerName
    precision: PrecisionName

    data_dir: str
    max_epochs: int
    batch_size: int
    limit_train_batches: float
    limit_val_batches: float

    @staticmethod
    def from_args(args: argparse.Namespace) -> TunerConfig:
        return TunerConfig(
            wandb_api_key=args.wandb_api_key,
            experiment_name=args.experiment_name,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            precision=args.precision,
            data_dir=args.data_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )


CLI_PARSER = argparse.ArgumentParser()

wandb_arg_group = CLI_PARSER.add_argument_group("WandB")
wandb_arg_group = CLI_PARSER.add_argument(
    "--wandb_api_key", type=str, required=True, help="Wandb API key"
)

experiment_arg_group = CLI_PARSER.add_argument_group("Experiment")
experiment_arg_group.add_argument(
    "--experiment_name", type=str, required=False, help="Name of the experiment"
)
experiment_arg_group.add_argument(
    "--optimizer",
    choices=get_args(OptimizerName),
    required=True,
    help="Optimizer to use",
)
experiment_arg_group.add_argument(
    "--scheduler",
    choices=get_args(SchedulerName),
    required=True,
    help="Scheduler to use",
)
experiment_arg_group.add_argument(
    "--precision",
    choices=get_args(PrecisionName),
    required=True,
    help="Precision to use",
)

data_loader_arg_group = CLI_PARSER.add_argument_group("DataLoader")
data_loader_arg_group.add_argument(
    "--data_dir", type=str, required=True, help="Path to data directory"
)
data_loader_arg_group.add_argument(
    "--max_epochs", type=int, required=True, help="Max number of epochs"
)
data_loader_arg_group.add_argument("--batch_size", type=int, required=True, help="Batch size")
data_loader_arg_group.add_argument(
    "--limit_train_batches", type=float, required=True, help="Limit train batches"
)
data_loader_arg_group.add_argument(
    "--limit_val_batches", type=float, required=True, help="Limit val batches"
)
