import math
from typing import Optional, overload
from torch import Tensor
from typing import Tuple
from metrics.l2 import L2


def compute_psnr(
    l2: L2, max_value: float, pred: Tensor, gt: Tensor, valid: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    # gt = gt.type_as(pred)
    # if valid is not None:
    #     valid = valid.type_as(pred)

    mse, ssim, lpips = l2(pred, gt, valid=valid)
    psnr = 20 * math.log10(max_value) - 10.0 * mse.log10()
    return psnr, ssim, lpips
