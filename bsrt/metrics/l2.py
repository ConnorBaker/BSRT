from __future__ import annotations
from dataclasses import field
from lpips import LPIPS
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)
from torchmetrics.metric import Metric
from typing import ClassVar
import torch
import torch.nn.functional as F


class L2(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    # FIXME: Why does LPIPS have unused model parameters when lpips=True (the default setting)?
    loss_fn: LPIPS = field(
        init=False, default_factory=lambda: LPIPS(net="alex", lpips=False)
    )

    # Losses
    mse: Tensor = field(init=False)
    ssim: Tensor = field(init=False)
    lpips: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, valid: Tensor | None = None) -> None:
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
            pred.type_as(gt).contiguous(),
            gt.contiguous(),
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
            data_range=1.0,
        )  # type: ignore
        lpips: Tensor = self.loss_fn(pred.contiguous(), gt.contiguous()).squeeze()

        self.mse: Tensor = mse
        self.ssim: Tensor = ssim
        self.lpips: Tensor = lpips

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.mse, self.ssim, self.lpips
