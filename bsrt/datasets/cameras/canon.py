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
class CanonImage:
    im_raw: Tensor
    metadata: ImageMetadata

    @staticmethod
    def load(path: Path) -> CanonImage:
        im_raw = cv2.imread((path / "im_raw.png").as_posix(), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)

        # FIXME: Will not be able to load this object and have it translated directly.
        metadata = pkl.load(open((path / "meta_info.pkl").as_posix(), "rb", -1))

        return CanonImage(im_raw.float(), ImageMetadata(**metadata.__dict__))

    def __post_init__(self) -> None:
        super().__init__()
        if (
            self.metadata.black_level is not None
            and len(self.metadata.black_level) == 4
        ):
            self.metadata.black_level.pop(2)

        if self.metadata.cam_wb is not None and len(self.metadata.cam_wb) == 4:
            self.metadata.cam_wb.pop(2)

        if (
            self.metadata.daylight_wb is not None
            and len(self.metadata.daylight_wb) == 4
        ):
            self.metadata.daylight_wb.pop(2)

        self.metadata.xyz_srgb_matrix = torch.tensor(
            [
                3.2404542,
                -1.5371385,
                -0.4985314,
                -0.9692660,
                1.8760108,
                0.0415560,
                0.0556434,
                -0.2040259,
                1.0572252,
            ]
        ).view(3, 3)

        self.metadata.norm_factor = 16383.0

    def shape(self) -> tuple[int, int, int]:
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

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
            im_raw -= torch.tensor(self.metadata.black_level).view(3, 1, 1)

        if white_balance:
            im_raw *= torch.tensor(self.metadata.cam_wb).view(3, 1, 1) / 1024.0

        if normalize:
            im_raw /= self.metadata.norm_factor

        return im_raw

    def get_crop(self, r1: int, r2: int, c1: int, c2: int) -> CanonImage:
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return CanonImage(
            im_raw,
            # Make a copy of the metadata.
            replace(self.metadata),
        )

    @overload
    def postprocess(self, return_np: Literal[False]) -> Tensor:
        ...

    @overload
    def postprocess(self, return_np: Literal[True]) -> npt.NDArray[np.uint8]:
        ...

    def postprocess(
        self, return_np: bool = True
    ) -> Union[Tensor, npt.NDArray[np.uint8]]:
        # Convert to rgb
        im = self.im_raw

        im = (
            im - torch.tensor(self.metadata.black_level).view(3, 1, 1)
        ).float() * torch.tensor(self.metadata.cam_wb).view(3, 1, 1)

        im_out = im / im.max()
        im_out = im_out.clamp(0.0, 1.0)

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out
