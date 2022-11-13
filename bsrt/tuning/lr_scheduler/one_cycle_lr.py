from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class OneCycleLRParams:
    max_lr: float
    epochs: int
    steps_per_epoch: int
    pct_start: float
    cycle_momentum: bool
    base_momentum: float
    max_momentum: float
    div_factor: float
    final_div_factor: float

    @classmethod
    def suggest(cls, trial: Trial, epochs: int, steps_per_epoch: int) -> OneCycleLRParams:
        base_momentum = trial.suggest_float("base_momentum", low=0.5, high=0.95)
        max_momentum = trial.suggest_float("max_momentum", low=base_momentum, high=1.0)
        return cls(
            max_lr=trial.suggest_float("max_lr", low=1e-3, high=1.0, log=True),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=trial.suggest_float("pct_start", low=0.0, high=1.0),
            cycle_momentum=trial.suggest_categorical(  # type: ignore
                "cycle_momentum", [True, False]
            ),
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=trial.suggest_float("div_factor", low=1e1, high=1e3, log=True),
            final_div_factor=trial.suggest_float("final_div_factor", low=1e3, high=1e5, log=True),
        )
