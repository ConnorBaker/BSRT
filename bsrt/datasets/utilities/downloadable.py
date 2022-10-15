from abc import ABC
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from typing_extensions import ClassVar


class Downloadable(ABC):
    url: ClassVar[str]
    filename: ClassVar[str]
    dirname: ClassVar[str]
    mirrors: ClassVar[list[str]]
    data_dir: str | Path

    def download(self) -> None:
        dir = Path(self.data_dir) / self.dirname
        file = Path(self.data_dir) / self.filename
        urls = [self.url] + self.mirrors

        match (dir.exists(), file.exists()):
            case (True, _):
                return
            case (False, True):
                extract_archive(from_path=file.as_posix(), to_path=dir.as_posix())
                return
            case (False, False):
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
                    except:
                        raise Exception("Could not download dataset")

                if not file.exists():
                    raise Exception("Could not download dataset")
