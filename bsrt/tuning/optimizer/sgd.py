from dataclasses import dataclass

from utilities import prepend_to_param_names

SGD_PARAMS = prepend_to_param_names(
    "SGD_PARAMS",
    [
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-6, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "momentum",
            "type": "range",
            "bounds": [1e-6, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "dampening",
            "type": "range",
            "bounds": [1e-6, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "wd_after_momentum",
            "type": "choice",
            "values": [True, False],
            "value_type": "bool",
            "is_ordered": True,
        },
        {
            "name": "weight_decay",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
    ],
)


@dataclass
class SGDParams:
    lr: float
    weight_decay: float
    dampening: float
    momentum: float
    wd_after_momentum: bool
