from dataclasses import dataclass
from typing import List

from ax import Parameter, ParameterType, RangeParameter


@dataclass
class DecoupledSGDWParams:
    lr: float
    momentum: float
    dampening: float
    weight_decay: float


DECOUPLED_SGDW_PARAMS: List[Parameter] = [
    RangeParameter(
        name="decoupled_sgdw_params.lr",
        parameter_type=ParameterType.FLOAT,
        lower=1e-6,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_sgdw_params.momentum",
        parameter_type=ParameterType.FLOAT,
        lower=1e-9,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_sgdw_params.dampening",
        parameter_type=ParameterType.FLOAT,
        lower=1e-9,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="decoupled_sgdw_params.weight_decay",
        parameter_type=ParameterType.FLOAT,
        lower=1e-9,
        upper=1e-3,
        log_scale=True,
    ),
]
