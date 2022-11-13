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
        flow_alignment_groups: int = trial.suggest_categorical(  # type: ignore
            "flow_alignment_groups", [2**i for i in range(2, 5)]
        )
        num_features = trial.suggest_int(
            "num_features",
            flow_alignment_groups,
            32 * flow_alignment_groups,
            step=flow_alignment_groups,
        )
        return cls(
            attn_drop_rate=trial.suggest_float("attn_drop_rate", low=0.0, high=1.0),
            drop_path_rate=trial.suggest_float("drop_path_rate", low=0.0, high=1.0),
            drop_rate=trial.suggest_float("drop_rate", low=0.0, high=1.0),
            mlp_ratio=trial.suggest_categorical(  # type: ignore
                "mlp_ratio", [2**i for i in range(1, 6)]
            ),
            flow_alignment_groups=flow_alignment_groups,
            num_features=num_features,
            qkv_bias=trial.suggest_categorical("qkv_bias", [True, False]),  # type: ignore
        )
