import math
from torch import Tensor
from metrics.l2 import L2


def compute_psnr(
    l2: L2, max_value: float, pred: Tensor, gt: Tensor, valid: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    mse, ssim, lpips = l2(pred, gt, valid=valid)
    psnr = 20 * math.log10(max_value) - 10.0 * mse.log10()
    return psnr, ssim, lpips
