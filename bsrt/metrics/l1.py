from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.metric import Metric
from metrics.utils.ignore_boundry import ignore_boundary


class L1(Metric):
    full_state_update = False

    def __init__(self, boundary_ignore: Optional[int] = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, valid: Optional[Tensor] = None) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        if valid is None:
            mse = F.l1_loss(pred, gt)
        else:
            mse = F.l1_loss(pred, gt, reduction="none")

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        self.mse: Tensor = mse

    def compute(self) -> Tensor:
        return self.mse
