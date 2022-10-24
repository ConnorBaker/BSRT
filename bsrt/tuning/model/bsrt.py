from dataclasses import dataclass
from typing import List

from ax import ChoiceParameter, Parameter, ParameterType, RangeParameter


@dataclass
class BSRTParams:
    attn_drop_rate: float
    drop_path_rate: float
    drop_rate: float
    mlp_ratio: float
    num_features: int
    qk_scale: float
    qkv_bias: bool


BSRT_PARAMS: List[Parameter] = [
    RangeParameter(
        name="bsrt_params.attn_drop_rate",
        parameter_type=ParameterType.FLOAT,
        lower=0.0,
        upper=1.0,
    ),
    RangeParameter(
        name="bsrt_params.drop_path_rate",
        parameter_type=ParameterType.FLOAT,
        lower=1e-2,
        upper=1.0,
        log_scale=True,
    ),
    RangeParameter(
        name="bsrt_params.drop_rate",
        parameter_type=ParameterType.FLOAT,
        lower=0.0,
        upper=1.0,
    ),
    ChoiceParameter(
        name="bsrt_params.mlp_ratio",
        parameter_type=ParameterType.FLOAT,
        values=[2.0**n for n in range(1, 5)],
        is_ordered=True,
        sort_values=False,
    ),
    ChoiceParameter(
        name="bsrt_params.num_features",
        parameter_type=ParameterType.INT,
        values=[2**n for n in range(4, 8)],
        is_ordered=True,
        sort_values=False,
    ),
    RangeParameter(
        name="bsrt_params.qk_scale",
        parameter_type=ParameterType.FLOAT,
        lower=0.0,
        upper=1.0,
    ),
    ChoiceParameter(
        name="bsrt_params.qkv_bias",
        parameter_type=ParameterType.BOOL,
        values=[True, False],
        is_ordered=True,
        sort_values=False,
    ),
]
