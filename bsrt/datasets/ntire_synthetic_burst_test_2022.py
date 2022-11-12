from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Union, _TypedDict

import torch
import torchvision
from torch import Tensor
from torchvision.datasets import VisionDataset
from typing_extensions import ClassVar

from bsrt.datasets.utilities.downloadable import Downloadable


# TODO: Do I need to normalize the images or convert them to floats?
# TODO: Document the type of the returned tensor.
@dataclass
class NTIRESyntheticBurstTest2022(VisionDataset, Downloadable):
    """Synthetic burst test set. The test burst have been generated using the same synthetic
    pipeline as employed in SyntheticBurst dataset.
    https://data.vision.ee.ethz.ch/bhatg/synburst_test_2022.zip

    Args:
        data_dir (str): Path to the directory where the dataset is stored.
        burst_size (int): Number of frames in the burst. Default: 14.
        transform (callable): A function/transform takes in a burst (a tensor of shape
            [burst_size, 4, 128, 128]) and returns a transformed version.
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/synburst_test_2022.zip"
    filename: ClassVar[str] = "ntire_synburst_test_2022.zip"
    dirname: ClassVar[str] = "ntire_synburst_test_2022"
    mirrors: ClassVar[List[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/synburst_test_2022.zip"
    ]

    data_dir: str
    burst_size: int = 14
    transform: Callable[[Tensor], Union[Tensor, _TypedDict]] = field(default=lambda x: x)

    def __getitem__(self, index: int) -> Union[Tensor, _TypedDict]:
        """
        Args:
            index (int): Index of the burst to be returned. Must be in the range [0, 92).

        Returns:
            burst (Tensor): A tensor of shape [burst_size, 4, 128, 128].

                The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer
                mosaick.
        """
        image_pngs: List[Tensor] = []
        for i in range(self.burst_size):
            image_path = Path(self.data_dir) / self.dirname / f"{index:04}" / f"im_raw_{i:02}.png"
            image_file = torchvision.io.read_file(image_path.as_posix())

            # TODO: Do we need to explicitly state the image read mode? These are bayered images
            # and we don't want them to be interpreted as RGB PNG files with an alpha layer.
            image_png = torchvision.io.decode_png(image_file)
            image_pngs.append(image_png)

        stacked = torch.stack(image_pngs)
        transformed = self.transform(stacked)
        return transformed

    def __len__(self) -> int:
        return 92
