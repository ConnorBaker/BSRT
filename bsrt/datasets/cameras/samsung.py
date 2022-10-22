from __future__ import annotations

import pickle as pkl
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union, overload

import cv2
import numpy as np
import numpy.typing as npt
import torch
from datasets.cameras.metadata import ImageMetadata
from torch import Tensor
from typing_extensions import Literal


@dataclass
class SamsungImage:
    im_raw: Tensor
    metadata: ImageMetadata

    @staticmethod
    def load(path: Path) -> SamsungImage:
        im_raw = cv2.imread((path / "im_raw.png").as_posix(), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)

        # FIXME: Will not be able to load this object and have it translated directly.
        metadata = pkl.load(open((path / "meta_info.pkl").as_posix(), "rb", -1))

        return SamsungImage(im_raw, ImageMetadata(**metadata.__dict__))

    def __post_init__(self) -> None:
        super().__init__()
        self.metadata.norm_factor = 1023.0

    def get_exposure_time(self):
        assert self.metadata.exif_data is not None
        return self.metadata.exif_data["EXIF ExposureTime"].values[0].decimal()

    def get_f_number(self):
        assert self.metadata.exif_data is not None
        return self.metadata.exif_data["EXIF FNumber"].values[0].decimal()

    def get_iso(self):
        assert self.metadata.exif_data is not None
        return self.metadata.exif_data["EXIF ISOSpeedRatings"].values[0]

    def get_image_data(
        self,
        substract_black_level: bool = False,
        white_balance: bool = False,
        normalize: bool = False,
    ) -> Tensor:
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw -= torch.tensor(self.metadata.black_level).view(4, 1, 1)

        if white_balance:
            im_raw *= torch.tensor(self.metadata.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw /= self.metadata.norm_factor

        return im_raw

    def get_crop(self, r1: int, r2: int, c1: int, c2: int) -> SamsungImage:
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        if self.metadata.im_preview is not None:
            im_preview = self.metadata.im_preview[2 * r1 : 2 * r2, 2 * c1 : 2 * c2]
        else:
            im_preview = None

        return SamsungImage(im_raw, replace(self.metadata, im_preview=im_preview))

    @overload
    def postprocess(
        self, return_np: Literal[False], norm_factor: Union[float, None] = None
    ) -> Tensor:
        ...

    @overload
    def postprocess(
        self, return_np: Literal[True], norm_factor: Union[float, None] = None
    ) -> npt.NDArray[np.uint8]:
        ...

    def postprocess(
        self, return_np: bool = True, norm_factor: Union[float, None] = None
    ) -> Union[Tensor, npt.NDArray[np.uint8]]:
        # Convert to rgb
        im = self.im_raw

        im = (
            im - torch.tensor(self.metadata.black_level).view(4, 1, 1)
        ) * torch.tensor(self.metadata.cam_wb).view(4, 1, 1)

        if norm_factor is None:
            im /= im.max()
        else:
            im /= norm_factor

        im = torch.stack((im[0], (im[1] + im[2]) / 2, im[3]), dim=0)
        im_out = im.clamp(0.0, 1.0)

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out
