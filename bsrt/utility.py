from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import get_args

from bsrt.utils.types import BayerPattern, NormalizationMode

######################## BayerUnifyAug ############################


def bayer_unify(
    raw: Tensor,
    input_pattern: BayerPattern,
    target_pattern: BayerPattern,
    mode: NormalizationMode,
) -> Tensor:
    """
    Convert a bayer raw image from one bayer pattern to another.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """

    if input_pattern == target_pattern:
        # A match!
        h_offset, w_offset = 0, 0
    elif input_pattern[0:2] == target_pattern[2:4]:
        # Channels are rotated
        h_offset, w_offset = 1, 0
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        # Channels are flipped
        h_offset, w_offset = 0, 1
    elif input_pattern[0:2] == target_pattern[3:1:-1]:
        # Channels are rotated and flipped
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError("Unexpected pair of input and target bayer pattern!")

    if mode == "pad":
        out = F.pad(raw, (w_offset, w_offset, h_offset, h_offset), mode="reflect")
    elif mode == "crop":
        _, _, _, h, w = raw.shape
        out = raw[..., h_offset : h - h_offset, w_offset : w - w_offset]

    return out


def bayer_aug(
    raw: Tensor,
    flip_h: bool = False,
    flip_w: bool = False,
    transpose: bool = False,
    input_pattern: BayerPattern = "RGGB",
) -> Tensor:
    """
    Apply augmentation to a bayer raw image.
    """

    aug_pattern: str = input_pattern
    target_pattern: BayerPattern = input_pattern

    out = raw
    if flip_h:
        out = torch.flip(out, [3])  # GBRG, RGGB
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out = torch.flip(out, [4])
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out = out.permute(0, 1, 2, 4, 3)
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

    assert aug_pattern in get_args(BayerPattern)
    aug_pattern = cast(BayerPattern, aug_pattern)
    out = bayer_unify(out, aug_pattern, target_pattern, "crop")
    return out
