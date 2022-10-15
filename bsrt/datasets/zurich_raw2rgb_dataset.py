from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torchvision
from datasets.utilities.downloadable import Downloadable
from torch import Tensor
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset
from typing_extensions import ClassVar


@dataclass
class ZurichRaw2RgbDataset(VisionDataset, Downloadable):
    """Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip"
    filename: ClassVar[str] = "zurich-raw-to-rgb.zip"
    dirname: ClassVar[str] = "zurich-raw-to-rgb"
    mirrors: ClassVar[list[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/zurich-raw-to-rgb.zip"
    ]

    data_dir: str
    transform: Callable[[Tensor], Tensor | dict[str, Tensor]] = field(
        default=lambda x: x
    )
    path_strs: list[str] = field(init=False)

    def __post_init__(self) -> None:
        dataset_dir = Path(self.data_dir) / self.dirname
        self.path_strs = [path.as_posix() for path in dataset_dir.rglob("*.jpg")]

    def __getitem__(self, index: int) -> Tensor | dict[str, Tensor]:
        image_file = torchvision.io.read_file(self.path_strs[index])
        # TODO: Use nvjpg to decode the images more quickly when on CUDA
        image_jpg = torchvision.io.decode_jpeg(image_file)
        transformed = self.transform(image_jpg)
        return transformed

    def __len__(self) -> int:
        return len(self.path_strs)
