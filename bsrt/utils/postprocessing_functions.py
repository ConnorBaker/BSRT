from abc import ABC
from dataclasses import dataclass
from typing import Union, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from typing_extensions import Literal

from ..data_processing.camera_pipeline import (
    apply_ccm,
    apply_smoothstep,
    gamma_compression,
)
from ..data_processing.synthetic_burst_generation import MetaInfo
from .data_format_utils import torch_to_npimage


@dataclass
class PostProcess(ABC):
    def process(self, image, meta_info):
        raise NotImplementedError()


@dataclass
class SimplePostProcess(PostProcess):
    gains: bool = True
    ccm: bool = True
    gamma: bool = True
    smoothstep: bool = True
    return_np: bool = False

    def process(self, image: Tensor, meta_info: MetaInfo):
        return process_linear_image_rgb(
            image,
            meta_info,
            self.gains,
            self.ccm,
            self.gamma,
            self.smoothstep,
            self.return_np,
        )


@overload
def process_linear_image_rgb(
    image: Tensor,
    meta_info: MetaInfo,
    gains: bool = True,
    ccm: bool = True,
    gamma: bool = True,
    smoothstep: bool = True,
    return_np: Literal[False] = False,
) -> Tensor:
    ...


@overload
def process_linear_image_rgb(
    image: Tensor,
    meta_info: MetaInfo,
    gains: bool = True,
    ccm: bool = True,
    gamma: bool = True,
    smoothstep: bool = True,
    return_np: Literal[True] = True,
) -> npt.NDArray[np.float32]:
    ...


def process_linear_image_rgb(
    image: Tensor,
    meta_info: MetaInfo,
    gains: bool = True,
    ccm: bool = True,
    gamma: bool = True,
    smoothstep: bool = True,
    return_np: bool = False,
) -> Union[Tensor, npt.NDArray[np.float32]]:
    if gains:
        image = meta_info.gains.apply(image)

    if ccm:
        image = apply_ccm(image, meta_info.cam2rgb)

    if meta_info.compress_gamma and gamma:
        image = gamma_compression(image)

    if meta_info.smoothstep and smoothstep:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        return torch_to_npimage(image)
    return image


@dataclass
class BurstSRPostProcess(PostProcess):
    no_white_balance: bool = False
    gamma: bool = True
    smoothstep: bool = True
    return_np: bool = False

    def process(
        self,
        image: Tensor,
        meta_info: MetaInfo,
        external_norm_factor: Union[float, None] = None,
    ):
        return process_burstsr_image_rgb(
            image,
            meta_info,
            external_norm_factor=external_norm_factor,
            no_white_balance=self.no_white_balance,
            gamma=self.gamma,
            smoothstep=self.smoothstep,
            return_np=self.return_np,
        )


def process_burstsr_image_rgb(
    im: Tensor,
    meta_info: MetaInfo,
    return_np: bool = False,
    external_norm_factor: Union[float, None] = None,
    gamma: bool = True,
    smoothstep: bool = True,
    no_white_balance: bool = False,
):
    im = im * meta_info.norm_factor

    if not meta_info.black_level_subtracted:
        assert meta_info.black_level is not None
        im = im - torch.tensor(meta_info.black_level)[[0, 1, -1]].view(3, 1, 1)

    if not meta_info.while_balance_applied and not no_white_balance:
        assert meta_info.cam_wb is not None
        im = im * (meta_info.cam_wb[[0, 1, -1]].view(3, 1, 1) / meta_info.cam_wb[1])

    im_out = im

    if external_norm_factor is None:
        im_out = im_out / im_out.max()
    else:
        im_out = im_out / external_norm_factor

    im_out = im_out.clamp(0.0, 1.0)

    if gamma:
        im_out = im_out ** (1.0 / 2.2)

    if smoothstep:
        # Smooth curve
        im_out = 3 * im_out**2 - 2 * im_out**3

    if return_np:
        im_out = im_out.permute(1, 2, 0).cpu().numpy() * 255.0
        im_out = im_out.astype(np.uint8)

    return im_out
