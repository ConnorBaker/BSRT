from dataclasses import dataclass
from typing import Tuple
from utilities import prepend_to_param_names

ADAM_PARAMS = prepend_to_param_names(
    "ADAM_PARAMS",
    [
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-6, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "beta_gradient",
            "type": "range",
            "bounds": [1e-5, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "beta_square",
            "type": "range",
            "bounds": [1e-4, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "eps",
            "type": "range",
            "bounds": [1e-9, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "weight_decay",
            "type": "choice",
            "values": [0.0, 1.0],
            "value_type": "float",
        },
    ],
)


@dataclass
class AdamParams:
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float
