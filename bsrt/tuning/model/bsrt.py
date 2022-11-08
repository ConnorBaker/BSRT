from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class BSRTParams:
    attn_drop_rate: float
    drop_path_rate: float
    drop_rate: float
    mlp_ratio: float
    num_features: int
    qk_scale: float
    qkv_bias: bool

    @classmethod
    def suggest(cls, trial: Trial) -> BSRTParams:
        return cls(
            attn_drop_rate=trial.suggest_float("attn_drop_rate", low=0.0, high=1.0),
            drop_path_rate=trial.suggest_float("drop_path_rate", low=1e-2, high=1.0, log=True),
            drop_rate=trial.suggest_float("drop_rate", low=0.0, high=1.0),
            mlp_ratio=trial.suggest_float("mlp_ratio", low=2.0, high=32.0, log=True),
            num_features=trial.suggest_int("num_features", low=16, high=256, log=True),
            qk_scale=trial.suggest_float("qk_scale", low=0.0, high=1.0),
            qkv_bias=trial.suggest_categorical("qkv_bias", [True, False]),  # type: ignore
        )
