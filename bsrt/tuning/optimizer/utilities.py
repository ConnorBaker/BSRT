from typing import Union

import torch.nn as nn
from composer.optim import DecoupledAdamW, DecoupledSGDW
from torch.optim.optimizer import Optimizer

from .decoupled_adamw import DecoupledAdamWParams
from .decoupled_sgdw import DecoupledSGDWParams


def configure_optimizer(
    model: nn.Module,
    optimizer_params: Union[DecoupledAdamWParams, DecoupledSGDWParams],
) -> Optimizer:
    if isinstance(optimizer_params, DecoupledAdamWParams):
        return DecoupledAdamW(
            model.parameters(),
            lr=optimizer_params.lr,
            betas=optimizer_params.betas,
            eps=optimizer_params.eps,
            weight_decay=optimizer_params.weight_decay,
            amsgrad=False,
        )
    elif isinstance(optimizer_params, DecoupledSGDWParams):
        return DecoupledSGDW(
            model.parameters(),
            lr=optimizer_params.lr,
            momentum=optimizer_params.momentum,
            dampening=optimizer_params.dampening,
            weight_decay=optimizer_params.weight_decay,
            nesterov=False,
        )
    else:
        raise ValueError(f"Unknown optimizer params: {optimizer_params}")
