from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Tuple, Union

import torch
from metrics.l2 import L2
from torch import Tensor
from torchmetrics.metric import Metric


# TODO: This serves as a catch-all loss function. We should split it into multiple loss functions, and use metrics provided by torchmetrics where possible.
# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False)
class PSNR(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: Union[int, None] = None
    max_value: float = 1.0
    # TODO: We cannot use the default factory with nn.Modules because we must call the super init before we can call the module init.
    l2: L2 = field(init=False)

    # Losses
    psnr: Tensor = field(init=False)
    ssim: Tensor = field(init=False)
    lpips: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.l2 = L2(boundary_ignore=self.boundary_ignore)
        self.add_state("psnr", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(
        self, pred: Tensor, gt: Tensor, valid: Union[Tensor, None] = None
    ) -> None:
        """
        Args:
            pred: (B, C, H, W)
            gt: (B, C, H, W)
            valid: (B, C, H, W)
        """
        assert (
            pred.dim() == 4
        ), f"pred must be a 4D tensor, actual shape was {pred.shape}"
        assert (
            pred.shape == gt.shape
        ), f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}"

        mse, ssim, lpips = self.l2(pred, gt, valid=valid)
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()
        self.psnr = psnr
        self.ssim = ssim
        self.lpips = lpips

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.psnr, self.ssim, self.lpips
