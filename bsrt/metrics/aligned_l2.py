from __future__ import annotations
from dataclasses import dataclass, field
from lpips import LPIPS
from metrics.utils.prepare_aligned import prepare_aligned
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)
from torchmetrics.metric import Metric
from typing import ClassVar
from utils.spatial_color_alignment import get_gaussian_kernel
import torch
import torch.nn.functional as F


@dataclass
class AlignedL2(Metric):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    sr_factor: int = 4
    boundary_ignore: int | None = None
    loss_fn: LPIPS = field(init=False, default_factory=lambda: LPIPS(net="alex"))
    gauss_kernel: Tensor = field(init=False)
    ksz: int = field(init=False)

    # Losses
    mse: Tensor = field(init=False)
    ssim: Tensor = field(init=False)
    lpips: Tensor = field(init=False)

    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __post_init__(self) -> None:
        super().__init__()
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

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.mse, self.ssim, self.lpips
