from __future__ import annotations
from dataclasses import dataclass, field
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)
from torchmetrics.metric import Metric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from typing import ClassVar
import torch
import torch.nn.functional as F


# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False)
class L2(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    # FIXME: Why does LPIPS have unused model parameters when lpips=True (the default setting)?
    # TODO: We cannot use the default factory with nn.Modules because we must call the super init before we can call the module init.
    loss_fn: LPIPS = field(init=False)

    # Losses
    mse: Tensor = field(init=False)
    ssim: Tensor = field(init=False)
    lpips: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.loss_fn = LPIPS(net="alex", lpips=False)
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, valid: Tensor | None = None) -> None:
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

        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        assert pred.device == gt.device and (
            (gt.device == valid.device) if valid is not None else True
        ), f"pred, gt, and valid must be on the same device"

        acc_mse: Tensor = torch.tensor(0.0, device=pred.device)
        acc_ssim: Tensor = torch.tensor(0.0, device=pred.device)
        acc_lpips: Tensor = torch.tensor(0.0, device=pred.device)
        for pred, gt, valid in zip(
            pred, gt, valid if valid is not None else [None] * len(pred)
        ):
            if valid is None:
                mse: Tensor = F.mse_loss(pred, gt)
            else:
                mse_tensor: Tensor = F.mse_loss(pred, gt, reduction="none")
                eps: float = 1e-12
                elem_ratio: float = mse_tensor.numel() / valid.numel()
                mse: Tensor = (mse_tensor * valid).sum() / (
                    valid.sum() * elem_ratio + eps
                )

            acc_mse += mse

            # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
            acc_ssim += compute_ssim(
                pred.type_as(gt).contiguous(),
                gt.contiguous(),
                gaussian_kernel=True,
                kernel_size=11,
                sigma=1.5,
                reduction="elementwise_mean",
                data_range=1.0,
            )  # type: ignore
            acc_lpips += self.loss_fn(pred.contiguous(), gt.contiguous()).squeeze()

        self.mse = acc_mse / len(pred)
        self.ssim = acc_ssim / len(pred)
        self.lpips = acc_lpips / len(pred)

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.mse, self.ssim, self.lpips
