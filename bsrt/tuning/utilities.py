from typing import Dict, List, TypeVar, Union, cast
from hyperparameter_tuning.optimizer.adam import AdamParams
from hyperparameter_tuning.optimizer.sgd import SGDParams
from apex.optimizers import FusedAdam, FusedSGD
import torch.nn as nn
from torch.optim.optimizer import Optimizer


_T = TypeVar("_T")


def prepend_to_param_names(
    prefix: str, params: List[Dict[str, _T]]
) -> List[Dict[str, _T]]:
    return [dict(param, name=cast(_T, f"{prefix}.{param['name']}")) for param in params]


def filter_and_remove_from_keys(prefix: str, params: Dict[str, _T]) -> Dict[str, _T]:
    return {
        k.replace(f"{prefix}.", "", 1): v
        for k, v in params.items()
        if k.startswith(prefix)
    }


def configure_optimizer(
    model: nn.Module,
    optimizer_params: Union[AdamParams, SGDParams],
    # lr_scheduler_params: SchedulerParams,
) -> Optimizer:
    if isinstance(optimizer_params, AdamParams):
        return FusedAdam(
            model.parameters(),
            bias_correction=True,
            adam_w_mode=True,
            amsgrad=False,
            set_grad_none=True,
            **optimizer_params.__dict__,
        )
    elif isinstance(optimizer_params, SGDParams):
        return FusedSGD(
            model.parameters(),
            set_grad_none=True,
            materialize_master_grads=True,
            **optimizer_params.__dict__,
        )
    else:
        raise ValueError(f"Unknown optimizer params: {optimizer_params}")

    # if isinstance(scheduler_params, CosineAnnealingLRParams):
    #     scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=scheduler_params.T_max,
    #         eta_min=scheduler_params.eta_min,
    #     )
    # elif isinstance(scheduler_params, CosineAnnealingWarmRestartsParams):
    #     scheduler = CosineAnnealingWarmRestarts(
    #         optimizer,
    #         T_0=scheduler_params.T_0,
    #         T_mult=scheduler_params.T_mult,
    #         eta_min=scheduler_params.eta_min,
    #     )
    # elif isinstance(scheduler_params, OneCycleLRParams):
    #     scheduler = OneCycleLR(
    #         optimizer,
    #         max_lr=scheduler_params.max_lr,
    #         total_steps=scheduler_params.total_steps,
    #         pct_start=scheduler_params.pct_start,
    #         anneal_strategy=scheduler_params.anneal_strategy,
    #         cycle_momentum=scheduler_params.cycle_momentum,
    #         base_momentum=scheduler_params.base_momentum,
    #         max_momentum=scheduler_params.max_momentum,
    #         div_factor=scheduler_params.div_factor,
    #         final_div_factor=scheduler_params.final_div_factor,
    #         last_epoch=scheduler_params.last_epoch,
    #     )
    # elif isinstance(scheduler_params, MultiStepLRParams):
    #     scheduler = MultiStepLR(
    #         optimizer,
    #         milestones=scheduler_params.milestones,
    #         gamma=scheduler_params.gamma,
    #     )
    # elif isinstance(scheduler_params, ExponentialLRParams):
    #     scheduler = ExponentialLR(
    #         optimizer,
    #         gamma=scheduler_params.gamma,
    #     )
    # elif isinstance(scheduler_params, ReduceLROnPlateauParams):
    #     scheduler = ReduceLROnPlateau(
    #         optimizer,
    #         mode=scheduler_params.mode,
    #         factor=scheduler_params.factor
