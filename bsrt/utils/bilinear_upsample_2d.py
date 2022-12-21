import torch
import torch.nn.functional as F
from torch import Tensor


# TODO: Remove when https://github.com/pytorch/pytorch/pull/80340 is merged
def bilinear_upsample_2d(
    t: Tensor,
    scale_factor: None | float = None,
    size: None | tuple[int, int] = None,
    align_corners: bool = False,
    recompute_scale_factor: None | bool = None,
) -> Tensor:
    orig_precision = t.dtype
    if orig_precision == torch.bfloat16:
        t = t.to(torch.float32)

    # Pyright complains about interpolate being partially unknown
    ret: Tensor = F.interpolate(  # type: ignore[attr-defined]
        t,
        scale_factor=scale_factor,
        size=size,
        mode="bilinear",
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    ret = ret.to(orig_precision)
    return ret
