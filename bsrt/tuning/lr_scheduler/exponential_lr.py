from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class ExponentialLRParams:
    gamma: float

    @classmethod
    def suggest(cls, trial: Trial) -> ExponentialLRParams:
        return cls(
            gamma=trial.suggest_float("gamma", low=1e-4, high=1.0, log=True),
        )
