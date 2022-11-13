from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class CosineAnnealingWarmRestartsParams:
    T_0: int
    T_mult: int
    eta_min: float

    @classmethod
    def suggest(cls, trial: Trial) -> CosineAnnealingWarmRestartsParams:
        return cls(
            T_0=trial.suggest_int("T_0", low=1, high=100),
            T_mult=trial.suggest_int("T_mult", low=1, high=10),
            eta_min=trial.suggest_float("eta_min", low=1e-5, high=1e-1, log=True),
        )
