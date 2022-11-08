from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List

import cv2
import numpy as np
import torch
import torchvision
from datasets.utilities.downloadable import Downloadable
from torch import Tensor
from torchvision.datasets import VisionDataset
from typing_extensions import ClassVar, TypedDict


@dataclass
class ValDataset(VisionDataset, Downloadable):
    """Synthetic burst validation set introduced in [1]. The validation burst have been generated using a
    synthetic data generation pipeline. The dataset can be downloaded from
    https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. CVPR 2021
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip"
    filename: ClassVar[str] = "SyntheticBurstVal.zip"
    dirname: ClassVar[str] = "SyntheticBurstVal"
    mirrors: ClassVar[List[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/SyntheticBurstVal.zip"
    ]

    data_dir: Path
    burst_list: List[int] = field(default_factory=lambda: list(range(300)))
    burst_size: int = 14

    def __post_init__(self) -> None:
        assert (
            1 <= self.burst_size and self.burst_size <= 14
        ), "Only burst size in [1,14] are supported (there are 14 images in the burst)"

    def _read_burst_image(self, index: int, image_id: int) -> Tensor:
        im_path = (
            Path(self.data_dir)
            / self.dirname
            / "bursts"
            / f"{index:04d}"
            / f"im_raw_{image_id:02d}.png"
        )
        im = cv2.imread(im_path.as_posix(), cv2.IMREAD_UNCHANGED)
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / 2**14
        return im_t

    def _read_gt_image(self, index: int) -> Tensor:
        gt_path = Path(self.data_dir) / self.dirname / "gt" / f"{index:04d}" / "im_rgb.png"
        gt = cv2.imread(gt_path.as_posix(), cv2.IMREAD_UNCHANGED)
        gt_t = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1).float() / 2**14
        return gt_t

    # def _read_meta_info(self, index: int) -> dict[str, Any]:
    #     meta_info_path = self.data_dir / "gt" / f"{index:04d}" / "meta_info.pkl"
    #     with open(meta_info_path.as_posix(), "rb") as input_file:
    #         meta_info = pkl.load(input_file)

    #     return meta_info

    def __getitem__(self, index: int) -> Tensor | dict[str, Tensor]:
        """Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   [14, 4, 48, 48]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image
        """
        burst_images = list(
            map(
                self._read_burst_image,
                [index] * self.burst_size,
                range(self.burst_size),
            )
        )
        gt = self._read_gt_image(index)

        burst = torch.stack(burst_images, 0)
        return {"burst": burst, "gt": gt}

    def __len__(self) -> int:
        return 300
