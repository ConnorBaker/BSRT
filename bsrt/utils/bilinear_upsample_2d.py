from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor


# TODO: Remove when https://github.com/pytorch/pytorch/pull/80340 is merged
def bilinear_upsample_2d(
    t: Tensor,
    scale_factor: Union[float, None] = None,
    size: Union[tuple[int, int], None] = None,
    align_corners: bool = False,
    recompute_scale_factor: Union[bool, None] = None,
) -> Tensor:
    orig_precision = t.dtype
    if orig_precision == torch.bfloat16:
        t = t.to(torch.float32)

    ret = F.interpolate(
        t,
        scale_factor=scale_factor,
        size=size,
        mode="bilinear",
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    ret = ret.to(orig_precision)
    return ret
