from dataclasses import dataclass


@dataclass
class BSRTParams:
    attn_drop_rate: float
    drop_path_rate: float
    drop_rate: float
    mlp_ratio: float
    num_features: int
    qk_scale: float
    qkv_bias: bool


BSRT_PARAMS = [
    {
        "name": "bsrt_params.attn_drop_rate",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
    },
    {
        "name": "bsrt_params.drop_path_rate",
        "type": "range",
        "bounds": [1e-2, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "bsrt_params.drop_rate",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
    },
    {
        "name": "bsrt_params.mlp_ratio",
        "type": "choice",
        "values": [2.0**n for n in range(1, 5)],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "bsrt_params.num_features",
        "type": "choice",
        "values": [2**n for n in range(4, 10)],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "bsrt_params.qk_scale",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
    },
    {
        "name": "bsrt_params.qkv_bias",
        "type": "choice",
        "values": [True, False],
        "value_type": "bool",
        "is_ordered": True,
    },
]
