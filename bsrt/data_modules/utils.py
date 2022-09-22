from pathlib import Path
from typing import List
from torchvision.datasets.utils import download_and_extract_archive, extract_archive


def prepare_data(
    data_dir: Path, filename: str, dirname: str, url: str, mirrors: List[str]
) -> None:
    dir = data_dir / dirname
    if not dir.exists():
        file = data_dir / filename
        if not file.exists():
            urls = [url] + mirrors
            while not len(urls) == 0:
                url = urls.pop()
                try:
                    download_and_extract_archive(
                        url,
                        download_root=data_dir.as_posix(),
                        extract_root=dir.as_posix(),
                        filename=filename,
                    )
                    break
                except:
                    print(f"Failed to download from {url}")
                    pass
            if not file.exists():
                raise Exception("Could not download dataset")
        else:
            extract_archive(from_path=file.as_posix(), to_path=dir.as_posix())
