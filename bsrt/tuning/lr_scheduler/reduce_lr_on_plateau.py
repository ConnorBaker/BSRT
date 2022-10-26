from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class ReduceLROnPlateauParams:
    factor: float

    @classmethod
    def suggest(cls, trial: Trial) -> ReduceLROnPlateauParams:
        return cls(
            factor=trial.suggest_float("factor", low=1e-4, high=1.0, log=True),
        )
