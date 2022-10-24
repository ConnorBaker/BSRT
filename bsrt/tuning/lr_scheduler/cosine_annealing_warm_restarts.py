from dataclasses import dataclass
from typing import List

from ax import Parameter, ParameterType, RangeParameter


@dataclass
class CosineAnnealingWarmRestartsParams:
    T_0: int
    T_mult: int
    eta_min: float


COSINE_ANNEALING_WARM_RESTARTS_PARAMS: List[Parameter] = [
    RangeParameter(
        name="cosine_annealing_warm_restarts_params.T_0",
        parameter_type=ParameterType.INT,
        lower=1,
        upper=1000,
    ),
    RangeParameter(
        name="cosine_annealing_warm_restarts_params.T_mult",
        parameter_type=ParameterType.INT,
        lower=1,
        upper=10,
    ),
    RangeParameter(
        name="cosine_annealing_warm_restarts_params.eta_min",
        parameter_type=ParameterType.FLOAT,
        lower=1e-9,
        upper=1e-3,
        log_scale=True,
    ),
]
