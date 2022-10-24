from __future__ import annotations

import argparse
from dataclasses import dataclass

from typing_extensions import Literal, get_args

OptimizerName = Literal["DecoupledAdamW", "DecoupledSGDW"]
SchedulerName = Literal[
    "CosineAnnealingWarmRestarts", "ExponentialLR", "ReduceLROnPlateau"
]
PrecisionName = Literal["bf16", 16, 32]


@dataclass
class TunerConfig:
    db_user: str
    db_pass: str
    db_host: str
    db_port: int
    db_name: str

    wandb_api_key: str

    experiment_name: str
    optimizer: OptimizerName
    scheduler: SchedulerName
    precision: PrecisionName

    num_trials: int
    max_epochs: int
    batch_size: int
    limit_train_batches: float
    limit_val_batches: float

    @staticmethod
    def from_args(args: argparse.Namespace) -> TunerConfig:
        return TunerConfig(
            db_user=args.db_user,
            db_pass=args.db_pass,
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            wandb_api_key=args.wandb_api_key,
            experiment_name=args.experiment_name,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            precision=args.precision,
            num_trials=args.num_trials,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )


CLI_PARSER = argparse.ArgumentParser()
CLI_PARSER.add_argument_group("Database")
CLI_PARSER.add_argument("--db_user", type=str, required=True, help="Database user")
CLI_PARSER.add_argument("--db_pass", type=str, required=True, help="Database password")
CLI_PARSER.add_argument("--db_host", type=str, required=True, help="Database host")
CLI_PARSER.add_argument("--db_port", type=int, required=True, help="Database port")
CLI_PARSER.add_argument("--db_name", type=str, required=True, help="Database name")


CLI_PARSER.add_argument_group("WandB")
CLI_PARSER.add_argument(
    "--wandb_api_key", type=str, required=True, help="Wandb API key"
)


CLI_PARSER.add_argument_group("Experiment")
CLI_PARSER.add_argument(
    "--experiment_name", type=str, required=True, help="Name of the experiment"
)
CLI_PARSER.add_argument(
    "--optimizer",
    choices=get_args(OptimizerName),
    required=True,
    help="Optimizer to use",
)
CLI_PARSER.add_argument(
    "--scheduler",
    choices=get_args(SchedulerName),
    required=True,
    help="Scheduler to use",
)
CLI_PARSER.add_argument(
    "--precision",
    choices=get_args(PrecisionName),
    required=True,
    help="Precision to use",
)


CLI_PARSER.add_argument_group("DataLoader")
CLI_PARSER.add_argument(
    "--num_trials", type=int, required=True, help="Number of trials"
)
CLI_PARSER.add_argument(
    "--max_epochs", type=int, required=True, help="Max number of epochs"
)
CLI_PARSER.add_argument("--batch_size", type=int, required=True, help="Batch size")
CLI_PARSER.add_argument(
    "--limit_train_batches", type=float, required=True, help="Limit train batches"
)
CLI_PARSER.add_argument(
    "--limit_val_batches", type=float, required=True, help="Limit val batches"
)
