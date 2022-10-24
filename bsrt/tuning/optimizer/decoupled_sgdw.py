from dataclasses import dataclass


@dataclass
class DecoupledSGDWParams:
    lr: float
    momentum: float
    dampening: float
    weight_decay: float


DECOUPLED_SGDW_PARAMS = [
    {
        "name": "decoupled_sgdw_params.lr",
        "type": "range",
        "bounds": [1e-6, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_sgdw_params.momentum",
        "type": "range",
        "bounds": [1e-9, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_sgdw_params.dampening",
        "type": "range",
        "bounds": [1e-9, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "decoupled_sgdw_params.weight_decay",
        "type": "range",
        "bounds": [1e-9, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
]
