from pathlib import Path
from cv2 import imread, Mat
from torch.utils.data import Dataset


class ZurichRaw2RgbDataset(Dataset):
    """Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    url = "https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip"
    filename = "zurich-raw-to-rgb.zip"
    dirname = "zurich-raw-to-rgb"
    mirrors = ["https://storage.googleapis.com/bsrt-supplemental/zurich-raw-to-rgb.zip"]

    def __init__(self, root: Path) -> None:
        self.root = root / "train" / "canon"
        self.image_list = list(self.root.glob("*.jpg"))[:1000]

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index) -> Mat:
        return imread(self.image_list[index].as_posix())
