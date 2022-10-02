from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import Tensor
from metrics.utils.ignore_boundry import ignore_boundary
from loss.Charbonnier import CharbonnierLoss as CBLoss
from torchmetrics.metric import Metric


@dataclass
class CharbonnierLoss(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    charbonnier_loss: CBLoss = field(
        init=False, default_factory=lambda: CBLoss(reduce=True)
    )

    # Losses
    loss: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
        self.loss: Tensor = self.charbonnier_loss(pred, gt)

    def compute(self) -> Tensor:
        return self.loss
