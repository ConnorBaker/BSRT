from dataclasses import dataclass
from typing import List

from ax import Parameter, ParameterType, RangeParameter


@dataclass
class ExponentialLRParams:
    gamma: float


EXPONENTIAL_LR_PARAMS: List[Parameter] = [
    RangeParameter(
        name="exponential_lr_params.gamma",
        parameter_type=ParameterType.FLOAT,
        lower=1e-4,
        upper=1.0,
        log_scale=True,
    )
]
