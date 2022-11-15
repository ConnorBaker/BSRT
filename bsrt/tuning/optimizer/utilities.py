from typing import Union

import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.optimizer import Optimizer

from bsrt.tuning.optimizer.adamw import AdamWParams
from bsrt.tuning.optimizer.sgd import SGDParams

OptimizerParams = Union[AdamWParams, SGDParams]


def configure_optimizer(model: nn.Module, optimizer_params: OptimizerParams) -> Optimizer:
    if isinstance(optimizer_params, AdamWParams):
        return AdamW(
            model.parameters(),
            lr=optimizer_params.lr,
            betas=(optimizer_params.beta_gradient, optimizer_params.beta_square),
            eps=optimizer_params.eps,
            weight_decay=optimizer_params.weight_decay,
            amsgrad=False,
        )
    elif isinstance(optimizer_params, SGDParams):
        return SGD(
            model.parameters(),
            lr=optimizer_params.lr,
            momentum=optimizer_params.momentum,
            dampening=optimizer_params.dampening,
            weight_decay=optimizer_params.weight_decay,
            nesterov=False,
        )
    else:
        raise ValueError(f"Unknown optimizer params: {optimizer_params}")
