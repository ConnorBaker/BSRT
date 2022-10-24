from dataclasses import dataclass
from typing import List

from ax import Parameter, ParameterType, RangeParameter


@dataclass
class ReduceLROnPlateauParams:
    factor: float


REDUCE_LR_ON_PLATEAU_PARAMS: List[Parameter] = [
    RangeParameter(
        name="reduce_lr_on_plateau_params.factor",
        parameter_type=ParameterType.FLOAT,
        lower=1e-4,
        upper=1.0,
        log_scale=True,
    )
]
