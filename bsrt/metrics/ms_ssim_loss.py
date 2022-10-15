from dataclasses import dataclass, field
from typing import ClassVar

import torch
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure as compute_msssim,
)
from torchmetrics.metric import Metric


# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False)
class MSSSIMLoss(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None

    # Losses
    loss: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
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
