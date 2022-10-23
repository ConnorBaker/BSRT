from dataclasses import dataclass
from typing import Tuple

ADAM_PARAMS = [
    {
        "name": "adam_params.lr",
        "type": "range",
        "bounds": [1e-6, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "adam_params.beta_gradient",
        "type": "range",
        "bounds": [1e-5, 1.0],
        "value_type": "float",
    },
    {
        "name": "adam_params.beta_square",
        "type": "range",
        "bounds": [1e-5, 1.0],
        "value_type": "float",
    },
    {
        "name": "adam_params.eps",
        "type": "range",
        "bounds": [1e-9, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "adam_params.weight_decay",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
    },
]

ADAM_PARAM_CONSTRAINTS = [
    "adam_params.beta_gradient <= adam_params.beta_square",
]


@dataclass
class AdamParams:
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float
