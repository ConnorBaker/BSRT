from dataclasses import dataclass

from utilities import prepend_to_param_names


@dataclass
class BSRTParams:
    attn_drop_rate: float
    drop_path_rate: float
    drop_rate: float
    mlp_ratio: float
    num_features: int
    qk_scale: float
    qkv_bias: bool


BSRT_PARAMS = prepend_to_param_names(
    "BSRT_PARAMS",
    [
        {
            "name": "attn_drop_rate",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "drop_path_rate",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "drop_rate",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "mlp_ratio",
            "type": "choice",
            "values": [2.0**n for n in range(1, 5)],
            "value_type": "float",
            "is_ordered": True,
        },
        {
            "name": "num_features",
            "type": "choice",
            "values": [2**n for n in range(4, 10)],
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "qk_scale",
            "type": "range",
            "bounds": [1e-8, 1.0],
            "value_type": "float",
        },
        {
            "name": "qkv_bias",
            "type": "choice",
            "values": [True, False],
            "value_type": "bool",
            "is_ordered": True,
        },
    ],
)
