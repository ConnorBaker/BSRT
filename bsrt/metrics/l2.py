from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple
from metrics.utils.ignore_boundry import ignore_boundary

# NOTE: We specifically do not use the LPIPS module torchmetrics ships with since it requires that all inputs are in the range [-1,1] and our SR outputs during training are regularly greater than one.
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)

from torchmetrics.metric import Metric
from lpips import LPIPS


class L2(Metric):
    full_state_update = False

    def __init__(self, boundary_ignore: Optional[int] = None):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.loss_fn = LPIPS(net="alex")
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, valid: Optional[Tensor] = None) -> None:
        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        if valid is None:
            mse = F.mse_loss(pred, gt)
        else:
            mse = F.mse_loss(pred, gt, reduction="none")

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
        ssim: Tensor = compute_ssim(
            pred.contiguous(),
            gt.contiguous(),
            # gaussian_kernel=True,
            # kernel_size=11,
            # sigma=1.5,
            # reduction="elementwise_mean",
            # data_range=1.0,
        )  # type: ignore
        lpips: Tensor = self.loss_fn(pred.contiguous(), gt.contiguous()).squeeze()

        self.mse: Tensor = mse
        self.ssim: Tensor = ssim
        self.lpips: Tensor = lpips

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.mse, self.ssim, self.lpips
