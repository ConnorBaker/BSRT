from abc import ABC
from pathlib import Path
from typing import List, Union

from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from typing_extensions import ClassVar


class Downloadable(ABC):
    url: ClassVar[str]
    filename: ClassVar[str]
    dirname: ClassVar[str]
    mirrors: ClassVar[List[str]]
    data_dir: Union[str, Path]

    def download(self) -> None:
        dir = Path(self.data_dir) / self.dirname
        file = Path(self.data_dir) / self.filename
        urls = [self.url] + self.mirrors

        if dir.exists():
            return
        elif file.exists():
            extract_archive(from_path=file.as_posix(), to_path=dir.as_posix())
            return
        else:
            while not len(urls) == 0:
                url = urls.pop()
                try:
                    download_and_extract_archive(
                        url,
                        download_root=Path(self.data_dir).as_posix(),
                        extract_root=dir.as_posix(),
                        filename=self.filename,
                    )
                    return
                except Exception as e:
                    raise Exception(f"Could not download dataset: {e}")

            if not file.exists():
                raise Exception("Could not download dataset")
