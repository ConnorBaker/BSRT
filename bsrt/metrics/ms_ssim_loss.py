from typing import Optional
import torch
from torch import Tensor
from metrics.utils.ignore_boundry import ignore_boundary
from torchmetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure as compute_msssim,
)
from torchmetrics.metric import Metric


class MSSSIMLoss(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)

    def __init__(self, boundary_ignore: Optional[int] = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
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
