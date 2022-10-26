from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class SGDParams:
    lr: float
    momentum: float
    dampening: float
    weight_decay: float

    @classmethod
    def suggest(cls, trial: Trial) -> SGDParams:
        return cls(
            lr=trial.suggest_float("lr", low=1e-6, high=1.0, log=True),
            momentum=trial.suggest_float("momentum", low=1e-9, high=1.0, log=True),
            dampening=trial.suggest_float("dampening", low=1e-9, high=1.0, log=True),
            weight_decay=trial.suggest_float(
                "weight_decay", low=1e-9, high=1e-3, log=True
            ),
        )
