from typing import Union

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torch.optim.optimizer import Optimizer

from bsrt.tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from bsrt.tuning.lr_scheduler.exponential_lr import ExponentialLRParams
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams


def configure_scheduler(
    optimizer: Optimizer,
    scheduler_params: Union[
        CosineAnnealingWarmRestartsParams, ExponentialLRParams, ReduceLROnPlateauParams
    ],
) -> _LRScheduler:
    if isinstance(scheduler_params, CosineAnnealingWarmRestartsParams):
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params.T_0,
            T_mult=scheduler_params.T_mult,
            eta_min=scheduler_params.eta_min,
        )
    elif isinstance(scheduler_params, ExponentialLRParams):
        return ExponentialLR(
            optimizer,
            gamma=scheduler_params.gamma,
        )
    elif isinstance(scheduler_params, ReduceLROnPlateauParams):
        return ReduceLROnPlateau(  # type: ignore
            optimizer,
            mode="min",  # We want to minimize the loss (LPIPS)
            factor=scheduler_params.factor,
        )
    else:
        raise ValueError(f"Unknown scheduler params: {scheduler_params}")
