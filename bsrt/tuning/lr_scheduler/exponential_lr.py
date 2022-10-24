from dataclasses import dataclass


@dataclass
class ExponentialLRParams:
    gamma: float


EXPONENTIAL_LR_PARAMS = [
    {
        "name": "exponential_lr_params.gamma",
        "type": "range",
        "bounds": [1e-4, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
]
