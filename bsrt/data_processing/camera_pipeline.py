from __future__ import annotations

from typing import overload

import cv2 as cv
import numpy as np
import numpy.typing as npt
import torch
from data_processing.meta_info import MetaInfo
from torch import Tensor
from typing_extensions import Literal
from utils.data_format_utils import torch_to_npimage

""" Based on http://timothybrooks.com/tech/unprocessing
Functions for forward and inverse camera pipeline. All functions input a torch float tensor of shape (c, h, w).
Additionally, some also support batch operations, i.e. inputs of shape (b, c, h, w)
"""


def random_ccm() -> Tensor:
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [
        [
            [1.0234, -0.2969, -0.2266],
            [-0.5625, 1.6328, -0.0469],
            [-0.0703, 0.2188, 0.6406],
        ],
        [
            [0.4913, -0.0541, -0.0202],
            [-0.613, 1.3513, 0.2906],
            [-0.1564, 0.2151, 0.7183],
        ],
        [
            [0.838, -0.263, -0.0639],
            [-0.2887, 1.0725, 0.2496],
            [-0.0627, 0.1427, 0.5438],
        ],
        [
            [0.6596, -0.2079, -0.0562],
            [-0.4782, 1.3016, 0.1933],
            [-0.097, 0.1581, 0.5181],
        ],
    ]

    num_ccms = len(xyz2cams)
    xyz2cams = torch.tensor(xyz2cams)

    weights = torch.FloatTensor(num_ccms, 1, 1).uniform_(0.0, 1.0)
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam /= rgb2cam.sum(dim=-1, keepdim=True)
    return rgb2cam


def apply_smoothstep(image: Tensor) -> Tensor:
    """Apply global tone mapping curve."""
    image_out = 3 * image**2 - 2 * image**3
    return image_out


def invert_smoothstep(image: Tensor) -> Tensor:
    """Approximately inverts a global tone mapping curve."""
    image = image.clamp(0.0, 1.0)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image: Tensor) -> Tensor:
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8) ** 2.2


def gamma_compression(image: Tensor) -> Tensor:
    """Converts from linear to gammaspace."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8) ** (1.0 / 2.2)


def apply_ccm(image: Tensor, ccm: Tensor) -> Tensor:
    """Applies a color correction matrix."""
    assert image.dim() == 3
    assert image.shape[0] == 3

    shape = image.shape
    image = image.view(3, -1)
    ccm = ccm.type_as(image)

    image = torch.mm(ccm, image)

    return image.view(shape)


def mosaic(image: Tensor, mode: Literal["grbg", "rggb"] = "rggb") -> Tensor:
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)

    match mode:
        case "rggb":
            red = image[:, 0, 0::2, 0::2]
            green_red = image[:, 1, 0::2, 1::2]
            green_blue = image[:, 1, 1::2, 0::2]
            blue = image[:, 2, 1::2, 1::2]
            image = torch.stack((red, green_red, green_blue, blue), dim=1)

        case "grbg":
            green_red = image[:, 1, 0::2, 0::2]
            red = image[:, 0, 0::2, 1::2]
            blue = image[:, 2, 0::2, 1::2]
            green_blue = image[:, 1, 1::2, 1::2]
            image = torch.stack((green_red, red, blue, green_blue), dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))


def demosaic(image: Tensor) -> Tensor:
    assert isinstance(image, torch.Tensor)
    image = image.clamp(0.0, 1.0) * 255

    if image.dim() == 4:
        num_images = image.dim()
        batch_input = True
    else:
        num_images = 1
        batch_input = False
        image = image.unsqueeze(0)

    # Generate single channel input for opencv
    im_sc = torch.zeros((num_images, image.shape[-2] * 2, image.shape[-1] * 2, 1))
    im_sc[:, ::2, ::2, 0] = image[:, 0, :, :]
    im_sc[:, ::2, 1::2, 0] = image[:, 1, :, :]
    im_sc[:, 1::2, ::2, 0] = image[:, 2, :, :]
    im_sc[:, 1::2, 1::2, 0] = image[:, 3, :, :]

    im_sc_np: npt.NDArray[np.uint8] = im_sc.numpy().astype(np.uint8)

    out: list[Tensor] = [
        df_utils.npimage_to_torch(
            cv.cvtColor(im, cv.COLOR_BAYER_BG2RGB), input_bgr=False
        )
        for im in im_sc_np
    ]

    if batch_input:
        return torch.stack(out, dim=0)
    else:
        return out[0]


@overload
def process_linear_image_rgb(
    image: Tensor, meta_info: MetaInfo, return_np: Literal[False]
) -> Tensor:
    ...


@overload
def process_linear_image_rgb(
    image: Tensor, meta_info: MetaInfo, return_np: Literal[True]
) -> npt.NDArray[np.float32]:
    ...


def process_linear_image_rgb(
    image: Tensor, meta_info: MetaInfo, return_np: bool = False
) -> Tensor | npt.NDArray[np.float32]:
    image = meta_info.gains.apply(image)
    image = apply_ccm(image, meta_info.cam2rgb)

    if meta_info.compress_gamma:
        image = gamma_compression(image)

    if meta_info.smoothstep:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        return torch_to_npimage(image)
    return image


def process_linear_image_raw(image: Tensor, meta_info: MetaInfo) -> Tensor:
    image = meta_info.gains.apply(image)
    image = demosaic(image)
    image = apply_ccm(image, meta_info.cam2rgb)

    if meta_info.compress_gamma:
        image = gamma_compression(image)

    if meta_info.smoothstep:
        image = apply_smoothstep(image)
    return image.clamp(0.0, 1.0)
