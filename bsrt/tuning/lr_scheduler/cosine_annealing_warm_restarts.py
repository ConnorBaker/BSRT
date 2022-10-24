from dataclasses import dataclass


@dataclass
class CosineAnnealingWarmRestartsParams:
    T_0: int
    T_mult: int
    eta_min: float


COSINE_ANNEALING_WARM_RESTARTS_PARAMS = [
    {
        "name": "cosine_annealing_warm_restarts_params.T_0",
        "type": "range",
        "bounds": [1, 1000],
        "value_type": "int",
    },
    {
        "name": "cosine_annealing_warm_restarts_params.T_mult",
        "type": "range",
        "bounds": [1, 10],
        "value_type": "int",
    },
    {
        "name": "cosine_annealing_warm_restarts_params.eta_min",
        "type": "range",
        "bounds": [1e-9, 1e-3],
        "value_type": "float",
        "log_scale": True,
    },
]
