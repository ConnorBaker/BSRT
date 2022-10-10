from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import Tensor
from metrics.utils.ignore_boundry import ignore_boundary
from loss.charbonnier import CharbonnierLoss as CBLoss
from torchmetrics.metric import Metric


# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False)
class CharbonnierLoss(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    # TODO: We cannot use the default factory with nn.Modules because we must call the super init before we can call the module init.
    charbonnier_loss: CBLoss = field(init=False)

    # Losses
    loss: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.charbonnier_loss = CBLoss(reduce=True)
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
        self.loss: Tensor = self.charbonnier_loss(pred, gt)

    def compute(self) -> Tensor:
        return self.loss
