from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.spatial_color_alignment import get_gaussian_kernel
from typing import Tuple
from metrics.utils.prepare_aligned import prepare_aligned

# NOTE: We specifically do not use the LPIPS module torchmetrics ships with since it requires that all inputs are in the range [-1,1] and our SR outputs during training are regularly greater than one.
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)

from torchmetrics.metric import Metric
from lpips import LPIPS


class AlignedL2(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __init__(
        self,
        alignment_net: nn.Module,
        sr_factor: int = 4,
        boundary_ignore: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net
        self.loss_fn = LPIPS(net="alex")
        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> None:
        pred_warped_m, gt, valid = prepare_aligned(
            alignment_net=self.alignment_net,
            pred=pred,
            gt=gt,
            burst_input=burst_input,
            kernel_size=self.ksz,
            gaussian_kernel=self.gauss_kernel,
            sr_factor=self.sr_factor,
            boundary_ignore=self.boundary_ignore,
        )

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m.contiguous(), gt.contiguous(), reduction="none")

        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
        ssim: Tensor = compute_ssim(
            pred_warped_m.type_as(gt).contiguous(),
            gt.contiguous(),
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
            data_range=1.0,
        )  # type: ignore

        lpips = self.loss_fn(pred_warped_m.contiguous(), gt.contiguous()).squeeze()

        self.mse: Tensor = mse
        self.ssim: Tensor = ssim
        self.lpips: Tensor = lpips

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.mse, self.ssim, self.lpips
