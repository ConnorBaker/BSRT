import os

# import urllib.request
# import zipfile
# import shutil
import argparse
import multiprocessing
from itertools import cycle
from torchvision.datasets.utils import download_and_extract_archive


def download_train_folders(out_dir: str, i: int) -> int:
    if not os.path.isfile(f"{out_dir}/train_{i:02d}.zip"):
        download_and_extract_archive(
            f"https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/train_{i:02d}.zip",
            download_root=out_dir,
        )
    return i


def download_burstsr_dataset(download_path):
    out_dir = download_path + "/burstsr_dataset"

    # Download train folders
    with multiprocessing.Pool() as pool:
        for i in pool.starmap(download_train_folders, zip(cycle([out_dir]), range(9))):
            print(f"Downloaded train_{i:02d}.zip")
        # end

    # # Unpack train set
    # for i in range(9):
    #     print("Unpacking train_{:02d}".format(i))
    #     with zipfile.ZipFile("{}/train_{:02d}.zip".format(out_dir, i), "r") as zip_ref:
    #         zip_ref.extractall("{}".format(out_dir))

    # # Move files to a common directory
    # os.makedirs("{}/train".format(out_dir), exist_ok=True)

    # for i in range(9):
    #     file_list = os.listdir("{}/train_{:02d}".format(out_dir, i))

    #     for b in file_list:
    #         source_dir = "{}/train_{:02d}/{}".format(out_dir, i, b)
    #         dst_dir = "{}/train/{}".format(out_dir, b)

    #         if os.path.isdir(source_dir):
    #             shutil.move(source_dir, dst_dir)

    # # Delete individual subsets
    # for i in range(9):
    #     shutil.rmtree("{}/train_{:02d}".format(out_dir, i))

    # # Unpack val set
    # print("Unpacking val")
    # with zipfile.ZipFile("{}/val.zip".format(out_dir), "r") as zip_ref:
    #     zip_ref.extractall("{}".format(out_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Downloads and unpacks BurstSR dataset"
    )
    parser.add_argument(
        "path", type=str, help="Path where the dataset will be downloaded"
    )

    args = parser.parse_args()

    download_burstsr_dataset(args.path)


if __name__ == "__main__":
    main()
