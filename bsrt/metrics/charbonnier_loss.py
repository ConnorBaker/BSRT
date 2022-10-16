from __future__ import annotations

from typing import ClassVar

import torch
from loss.charbonnier import CharbonnierLoss as CBLoss
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.metric import Metric


class CharbonnierLoss(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    charbonnier_loss: CBLoss

    # Losses
    loss: Tensor

    def __init__(self, boundary_ignore: int | None = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.charbonnier_loss = CBLoss(reduce=True)
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        self.loss: Tensor = self.charbonnier_loss(pred, gt)

    def compute(self) -> Tensor:
        return self.loss
