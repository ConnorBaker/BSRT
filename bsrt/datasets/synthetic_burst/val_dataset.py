from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from typing_extensions import TypedDict
import torch
from torch import Tensor
import cv2
import numpy as np
import pickle as pkl
from datasets.utilities.provides import ProvidesDatasetPipeline, ProvidesDatasource
from datasets.utilities.downloadable import Downloadable


class ValData(TypedDict):
    burst: Tensor
    gt: Tensor
    meta_info: dict[str, Any]


import ray
from ray.air import session
from ray.data import DatasetPipeline, Dataset
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig, DatasetConfig


@dataclass
class ValDataset(Downloadable, ProvidesDatasetPipeline):
    """Synthetic burst validation set introduced in [1]. The validation burst have been generated using a
    synthetic data generation pipeline. The dataset can be downloaded from
    https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. CVPR 2021
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip"
    filename: ClassVar[str] = "SyntheticBurstVal.zip"
    dirname: ClassVar[str] = "SyntheticBurstVal"
    mirrors: ClassVar[list[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/SyntheticBurstVal.zip"
    ]

    data_dir: Path
    burst_list: list[int] = field(default_factory=lambda: list(range(300)))
    burst_size: int = 14

    def dataset_pipeline() -> DatasetPipeline:

        pass

    def _read_burst_image(self, index: int, image_id: int) -> Tensor:

        pipe = ds.window(blocks_per_window=20)
        pipe = pipe.map(lambda obj: obj["image"])
        pipe = pipe.map(lambda ndarr: torch.from_numpy(ndarr._tensor))

        im_path = (
            self.data_dir / "bursts" / f"{index:04d}" / f"im_raw_{image_id:02d}.png"
        )
        im = cv2.imread(
            im_path.as_posix(),
            cv2.IMREAD_UNCHANGED,
        )
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (
            2**14
        )

        return im_t

    def _read_gt_image(self, index: int) -> Tensor:
        gt_path = self.data_dir / "gt" / f"{index:04d}" / "im_rgb.png"
        gt = cv2.imread(
            gt_path.as_posix(),
            cv2.IMREAD_UNCHANGED,
        )
        gt_t = (
            (torch.from_numpy(gt.astype(np.float32)) / 2**14).permute(2, 0, 1).float()
        )
        return gt_t

    def _read_meta_info(self, index: int) -> dict[str, Any]:
        meta_info_path = self.data_dir / "gt" / f"{index:04d}" / "meta_info.pkl"
        with open(meta_info_path.as_posix(), "rb") as input_file:
            meta_info = pkl.load(input_file)

        return meta_info

    def __getitem__(self, index: int) -> ValData:
        """Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   [14, 4, 48, 48]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image
            meta_info: Meta info about the burst which can be used to convert gt to sRGB space
        """
        burst_name = f"{index:04d}"
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        gt = self._read_gt_image(index)
        meta_info = self._read_meta_info(index)
        meta_info["burst_name"] = burst_name
        return ValData(burst=burst, gt=gt, meta_info=meta_info)
