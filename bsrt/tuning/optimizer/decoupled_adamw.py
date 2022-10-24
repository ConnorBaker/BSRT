from dataclasses import dataclass
from typing import Tuple


@dataclass
class DecoupledAdamWParams:
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


DECOUPLED_ADAMW_PARAMS = [
    {
        "name": "decoupled_adamw_params.lr",
        "type": "range",
        "bounds": [1e-6, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_adamw_params.beta_gradient",
        "type": "range",
        "bounds": [1e-3, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_adamw_params.beta_square",
        "type": "range",
        "bounds": [1e-3, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_adamw_params.eps",
        "type": "range",
        "bounds": [1e-9, 1e-7],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_adamw_params.weight_decay",
        "type": "range",
        "bounds": [1e-9, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
]
