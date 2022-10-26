from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from optuna.trial import Trial


@dataclass
class AdamWParams:
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float

    @classmethod
    def suggest(cls, trial: Trial) -> AdamWParams:
        return cls(
            lr=trial.suggest_float("lr", low=1e-6, high=1.0, log=True),
            betas=(
                trial.suggest_float("betas_gradient", low=1e-3, high=1.0, log=True),
                trial.suggest_float("betas_square", low=1e-3, high=1.0, log=True),
            ),
            eps=trial.suggest_float("eps", low=1e-9, high=1e-7, log=True),
            weight_decay=trial.suggest_float(
                "weight_decay", low=1e-9, high=1e-3, log=True
            ),
        )
