import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, List

# TODO: Remove when https://github.com/pytorch/pytorch/pull/80340 is merged
def bilinear_upsample_2d(t: Tensor, scale_factor: Optional[float] = None, size: Union[None, int, List[int]] = None, align_corners: Optional[bool] = False, recompute_scale_factor: Optional[bool] = None) -> Tensor:
    orig_precision = t.dtype
    if orig_precision == torch.bfloat16:
        t = t.to(torch.float16)
    
    ret = F.interpolate(t, scale_factor=scale_factor, size=size, mode="bilinear", align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)

    ret = ret.to(orig_precision)
    return ret