from typing import ClassVar

import torch
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure as compute_msssim,
)
from torchmetrics.metric import Metric


class MSSSIMLoss(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None

    # Losses
    loss: Tensor

    def __init__(self, boundary_ignore: int | None = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        self.loss: Tensor = compute_msssim(
            pred.type_as(gt).contiguous(),
            gt.contiguous(),
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
            data_range=1.0,
            normalize="relu",
        )

    def compute(self) -> Tensor:
        return self.loss
