from dataclasses import dataclass

SGD_PARAMS = [
    {
        "name": "sgd_params.lr",
        "type": "range",
        "bounds": [1e-6, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "sgd_params.momentum",
        "type": "range",
        "bounds": [1e-6, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "sgd_params.dampening",
        "type": "range",
        "bounds": [1e-6, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "sgd_params.wd_after_momentum",
        "type": "choice",
        "values": [True, False],
        "value_type": "bool",
        "is_ordered": True,
    },
    {
        "name": "sgd_params.weight_decay",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
    },
]


@dataclass
class SGDParams:
    lr: float
    weight_decay: float
    dampening: float
    momentum: float
    wd_after_momentum: bool
