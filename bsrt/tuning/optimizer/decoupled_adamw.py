from dataclasses import dataclass
from typing import List, Tuple

from ax import Parameter, ParameterType, RangeParameter


@dataclass
class DecoupledAdamWParams:
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


DECOUPLED_ADAMW_PARAMS: List[Parameter] = [
    RangeParameter(
        name="decoupled_adamw_params.lr",
        parameter_type=ParameterType.FLOAT,
        lower=1e-6,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_adamw_params.beta_gradient",
        parameter_type=ParameterType.FLOAT,
        lower=1e-3,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_adamw_params.beta_square",
        parameter_type=ParameterType.FLOAT,
        lower=1e-3,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_adamw_params.eps",
        parameter_type=ParameterType.FLOAT,
        lower=1e-9,
        upper=1e-7,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_adamw_params.weight_decay",
        parameter_type=ParameterType.FLOAT,
        lower=1e-9,
        upper=1e-3,
        log_scale=True,
    ),
]
