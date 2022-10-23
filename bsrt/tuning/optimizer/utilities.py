from typing import Union

import torch.nn as nn
from adam import AdamParams
from apex.optimizers import FusedAdam, FusedSGD
from sgd import SGDParams
from torch.optim.optimizer import Optimizer


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
