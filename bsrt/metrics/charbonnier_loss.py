from __future__ import annotations
import torch
from torch import Tensor
from metrics.utils.ignore_boundry import ignore_boundary
from loss.Charbonnier import CharbonnierLoss as CBLoss
from torchmetrics.metric import Metric


class CharbonnierLoss(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __init__(self, boundary_ignore: int | None = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.charbonnier_loss = CBLoss(reduce=True)
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
        self.loss: Tensor = self.charbonnier_loss(pred, gt)

    def compute(self) -> Tensor:
        return self.loss
