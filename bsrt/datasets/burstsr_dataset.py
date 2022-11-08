import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import Literal

from bsrt.datasets.cameras.canon import CanonImage
from bsrt.datasets.cameras.samsung import SamsungImage
from bsrt.datasets.utilities.utilities import flatten_raw_image, pack_raw_image


@dataclass
class BurstSRInfo:
    size: int
    path: Path


@dataclass
class BurstSRDataset(Dataset):
    """Real-world burst super-resolution dataset."""

    """
    args:
        data_dir : path of the data_dir directory
        burst_size : Burst size. Maximum allowed burst size is 14.
        crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
        center_crop: Whether to extract a random crop, or a centered crop.
        random_flip: Whether to apply random horizontal and vertical flip
        split: Can be 'train' or 'val'
    """
    data_dir: Path
    burst_list: List[Path] = field(init=False)
    burst_size: int = 8
    crop_sz: int = 80
    center_crop: bool = False
    random_flip: bool = False
    split: Literal["test", "train", "val"] = "train"
    substract_black_level: ClassVar[bool] = True
    white_balance: ClassVar[bool] = False

    def __post_init__(self) -> None:
        assert self.burst_size <= 14, "burst_sz must be less than or equal to 14"
        assert self.crop_sz <= 80, "crop_sz must be less than or equal to 80"
        assert self.split in ["test", "train", "val"]
        super().__init__()
        self.burst_list = sorted((self.data_dir / self.split).iterdir())

    def get_burst_info(self, burst_id: int) -> BurstSRInfo:
        return BurstSRInfo(size=self.burst_size, path=self.burst_list[burst_id])

    def _get_raw_image(self, burst_id: int, im_id: int) -> SamsungImage:
        raw_image = SamsungImage.load(self.burst_list[burst_id] / f"samsung_{im_id:02d}")
        return raw_image

    def _get_gt_image(self, burst_id: int) -> CanonImage:
        canon_im = CanonImage.load(self.burst_list[burst_id] / "canon")
        return canon_im

    @overload
    def get_burst(
        self,
        split: Literal["test"],
        burst_id: int,
        im_ids: List[int],
        info: Union[BurstSRInfo, None] = None,
    ) -> Tuple[List[SamsungImage], BurstSRInfo]:
        ...

    @overload
    def get_burst(
        self,
        split: Literal["train", "val"],
        burst_id: int,
        im_ids: List[int],
        info: Union[BurstSRInfo, None] = None,
    ) -> Tuple[List[SamsungImage], CanonImage, BurstSRInfo]:
        ...

    def get_burst(
        self,
        split: Literal["train", "test", "val"],
        burst_id: int,
        im_ids: List[int],
        info: Union[BurstSRInfo, None] = None,
    ) -> Union[
        Tuple[List[SamsungImage], BurstSRInfo],
        Tuple[List[SamsungImage], CanonImage, BurstSRInfo],
    ]:
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        if info is None:
            info = self.get_burst_info(burst_id)

        if split == "test":
            return frames, info
        else:
            gt = self._get_gt_image(burst_id)
            return frames, gt, info

    def _sample_images(self) -> List[int]:
        ids = random.sample(range(1, self.burst_size), k=self.burst_size - 1)
        ids = [
            0,
        ] + ids
        return ids

    def __len__(self) -> int:
        return len(self.burst_list)

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[Tensor, Dict[str, Any]],
        Tuple[Tensor, Tensor, Dict[str, Any], Dict[str, Any]],
    ]:
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids: List[int] = self._sample_images()

        # Read the burst images along with HR ground truth, if available
        if self.split == "test":
            frames, meta_info = self.get_burst(self.split, index, im_ids)
        else:
            frames, gt, meta_info = self.get_burst(self.split, index, im_ids)

        # Extract crop if needed
        if frames[0].shape()[-1] != self.crop_sz:
            if self.center_crop:
                r1: int = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
                c1: int = random.randint(0, frames[0].shape()[-1] - self.crop_sz)

            else:
                r1: int = (frames[0].shape()[-2] - self.crop_sz) // 2
                c1: int = (frames[0].shape()[-1] - self.crop_sz) // 2

            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

            if self.split != "test":
                scale_factor: int = gt.shape()[-1] // frames[0].shape()[-1]
                gt = gt.get_crop(
                    scale_factor * r1,
                    scale_factor * r2,
                    scale_factor * c1,
                    scale_factor * c2,
                )

        # Load the RAW image data
        burst_image_data = [
            im.get_image_data(
                normalize=True,
                substract_black_level=self.substract_black_level,
                white_balance=self.white_balance,
            )
            for im in frames
        ]

        # Convert to tensor
        if self.split != "test":
            gt_image_data = gt.get_image_data(
                normalize=True,
                white_balance=self.white_balance,
                substract_black_level=self.substract_black_level,
            )

        if self.random_flip:
            burst_image_data = [flatten_raw_image(im) for im in burst_image_data]

            pad = [0, 0, 0, 0]
            if random.random() > 0.5:
                burst_image_data = [
                    im.flip(
                        [
                            1,
                        ]
                    )[:, 1:-1].contiguous()
                    for im in burst_image_data
                ]
                if self.split != "test":
                    gt_image_data = gt_image_data.flip(
                        [
                            2,
                        ]
                    )[:, :, 2:-2].contiguous()

                pad[1] = 1

            if random.random() > 0.5:
                burst_image_data = [
                    im.flip(
                        [
                            0,
                        ]
                    )[1:-1, :].contiguous()
                    for im in burst_image_data
                ]
                if self.split != "test":
                    gt_image_data = gt_image_data.flip(
                        [
                            1,
                        ]
                    )[:, 2:-2, :].contiguous()

                pad[3] = 1

            burst_image_data = [pack_raw_image(im) for im in burst_image_data]
            burst_image_data = [
                F.pad(im.unsqueeze(0), pad, mode="replicate").squeeze(0) for im in burst_image_data
            ]

            if self.split != "test":
                gt_image_data = F.pad(
                    gt_image_data.unsqueeze(0), [4 * p for p in pad], mode="replicate"
                ).squeeze(0)

        burst_image_meta_info = frames[0].get_all_meta_data()

        burst_image_meta_info["black_level_subtracted"] = self.substract_black_level
        burst_image_meta_info["while_balance_applied"] = self.white_balance
        burst_image_meta_info["norm_factor"] = frames[0].norm_factor

        burst = torch.stack(burst_image_data, dim=0)
        burst_exposure = frames[0].get_exposure_time()
        burst_f_number = frames[0].get_f_number()
        burst_iso = frames[0].get_iso()
        light_factor_burst = burst_exposure * burst_iso / (burst_f_number**2)

        burst_image_meta_info["exposure"] = burst_exposure
        burst_image_meta_info["f_number"] = burst_f_number
        burst_image_meta_info["iso"] = burst_iso

        burst = burst.float()
        meta_info_burst = burst_image_meta_info

        for k, v in meta_info_burst.items():
            if isinstance(v, (list, tuple)):
                meta_info_burst[k] = torch.tensor(v)

        meta_info_burst["burst_name"] = meta_info["burst_name"]

        if self.split != "test":
            gt_image_meta_info = gt.get_all_meta_data()

            canon_exposure = gt.get_exposure_time()
            canon_f_number = gt.get_f_number()
            canon_iso = gt.get_iso()

            # Normalize the GT image to account for differences in exposure, ISO etc
            light_factor_canon = canon_exposure * canon_iso / (canon_f_number**2)
            exp_scale_factor = light_factor_burst / light_factor_canon
            gt_image_meta_info["exposure"] = canon_exposure
            gt_image_meta_info["f_number"] = canon_f_number
            gt_image_meta_info["iso"] = canon_iso

            gt_image_data = gt_image_data * exp_scale_factor
            gt_image_meta_info["black_level_subtracted"] = self.substract_black_level
            gt_image_meta_info["while_balance_applied"] = self.white_balance
            gt_image_meta_info["norm_factor"] = gt.norm_factor / exp_scale_factor

            frame_gt = gt_image_data.float()
            meta_info_gt = gt_image_meta_info

            for k, v in meta_info_gt.items():
                if isinstance(v, (list, tuple)):
                    meta_info_gt[k] = torch.tensor(v)

            return burst, frame_gt, meta_info_burst, meta_info_gt

        return burst, meta_info_burst
