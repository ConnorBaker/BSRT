from pathlib import Path
from typing import Any, list, TypedDict
import torch
from torch import Tensor
import cv2
import numpy as np
from torch.utils.data import Dataset


class TestData(TypedDict):
    burst: Tensor
    meta_info: dict[str, Any]


class TestDataset(Dataset):
    """Synthetic burst test set. The test burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """

    url: str = "https://data.vision.ee.ethz.ch/bhatg/synburst_test_2022.zip"
    filename: str = "synburst_test_2022.zip"
    dirname: str = "synburst_test_2022"
    mirrors: list[str] = [
        "https://storage.googleapis.com/bsrt-supplemental/synburst_test_2022.zip"
    ]

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.burst_list: list[int] = list(range(92))
        self.burst_size: int = 14

    def __len__(self) -> int:
        return len(self.burst_list)

    def _read_burst_image(self, index: int, image_id: int):
        im = cv2.imread(
            self.root / f"{index:04d}" / "im_raw_{image_id:02d}.png",
            cv2.IMREAD_UNCHANGED,
        )
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (
            2**14
        )
        return im_t

    def __getitem__(self, index: int) -> TestData:
        """Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            meta_info: Meta information about the burst
        """
        burst_name = f"{index:04d}"
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        return TestData(burst=burst, meta_info={"burst_name": burst_name})
