from pathlib import Path
from typing import Any, Dict, Tuple
import torch
from torch import Tensor
import cv2
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset


class SyntheticBurstVal(Dataset):
    """Synthetic burst validation set introduced in [1]. The validation burst have been generated using a
    synthetic data generation pipeline. The dataset can be downloaded from
    https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. CVPR 2021
    """

    url = "https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip"
    filename = "SyntheticBurstVal.zip"
    dirname = "SyntheticBurstVal"
    mirrors = ["https://storage.googleapis.com/bsrt-supplemental/SyntheticBurstVal.zip"]

    def __init__(self, root: Path) -> None:
        """
        args:
            root - Path to root dataset directory
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        self.root = root
        self.burst_list = list(range(300))
        self.burst_size = 14

    def __len__(self) -> int:
        return len(self.burst_list)

    def _read_burst_image(self, index: int, image_id: int) -> Tensor:
        im = cv2.imread(
            (self.root / "bursts" / f"{index:04d}" / f"{image_id:02d}.png").as_posix(),
            cv2.IMREAD_UNCHANGED,
        )
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (
            2**14
        )

        return im_t

    def _read_gt_image(self, index: int) -> Tensor:
        gt = cv2.imread(
            (self.root / f"gt/{index:04d}/im_rgb.png").as_posix(), cv2.IMREAD_UNCHANGED
        )
        gt_t = (
            (torch.from_numpy(gt.astype(np.float32)) / 2**14).permute(2, 0, 1).float()
        )
        return gt_t

    def _read_meta_info(self, index: int) -> Dict[str, Any]:
        with open(
            (self.root / f"gt/{index:04d}/meta_info.pkl").as_posix(), "rb"
        ) as input_file:
            meta_info = pkl.load(input_file)

        return meta_info

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
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
        return burst, gt, meta_info
