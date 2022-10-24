from dataclasses import dataclass


@dataclass
class ReduceLROnPlateauParams:
    factor: float


REDUCE_LR_ON_PLATEAU_PARAMS = [
    {
        "name": "reduce_lr_on_plateau_params.factor",
        "type": "range",
        "bounds": [1e-4, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
]
