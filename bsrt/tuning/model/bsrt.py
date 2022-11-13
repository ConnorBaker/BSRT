from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import Trial


@dataclass
class BSRTParams:
    attn_drop_rate: float
    drop_path_rate: float
    drop_rate: float
    mlp_ratio: float
    flow_alignment_groups: int
    num_features: int
    qkv_bias: bool

    @classmethod
    def suggest(cls, trial: Trial) -> BSRTParams:
        # num_features must be a multiple of flow_alignmnet_groups
        flow_alignment_groups = trial.suggest_int("flow_alignment_groups", 1, 32)
        num_features = trial.suggest_int(
            "num_features",
            flow_alignment_groups,
            32 * flow_alignment_groups,
            step=flow_alignment_groups,
        )
        return cls(
            attn_drop_rate=trial.suggest_float("attn_drop_rate", low=0.0, high=1.0),
            drop_path_rate=trial.suggest_float("drop_path_rate", low=1e-2, high=1.0, log=True),
            drop_rate=trial.suggest_float("drop_rate", low=0.0, high=1.0),
            mlp_ratio=trial.suggest_float("mlp_ratio", low=2.0, high=32.0, log=True),
            flow_alignment_groups=flow_alignment_groups,
            num_features=num_features,
            qkv_bias=trial.suggest_categorical("qkv_bias", [True, False]),  # type: ignore
        )
