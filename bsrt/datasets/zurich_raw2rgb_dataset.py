from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import torchvision
from torch import Tensor
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset
from typing_extensions import ClassVar

from .utilities.downloadable import Downloadable


@dataclass
class ZurichRaw2RgbDataset(VisionDataset, Downloadable):
    """Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip"
    filename: ClassVar[str] = "zurich-raw-to-rgb.zip"
    dirname: ClassVar[str] = "zurich-raw-to-rgb"
    mirrors: ClassVar[List[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/zurich-raw-to-rgb.zip"
    ]

    data_dir: str
    transform: Callable[[Tensor], Union[Tensor, Dict[str, Tensor]]] = field(
        default=lambda x: x
    )

    def __getitem__(self, index: int) -> Union[Tensor, Dict[str, Tensor]]:
        image_path = (
            Path(self.data_dir) / self.dirname / "train" / "canon" / f"{index}.jpg"
        )
        image_file = torchvision.io.read_file(image_path.as_posix())
        image_jpg = torchvision.io.decode_jpeg(image_file)
        transformed = self.transform(image_jpg)
        return transformed

    def __len__(self) -> int:
        return 46839
