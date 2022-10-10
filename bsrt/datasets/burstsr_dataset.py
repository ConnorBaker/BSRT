from dataclasses import field
import os
import torch
import cv2
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from datasets.cameras.samsung import SamsungImage
from datasets.cameras.canon import CanonImage
from datasets.utilities.utilities import flatten_raw_image, pack_raw_image

class BurstSRDataset(Dataset):
    """Real-world burst super-resolution dataset."""

    def __init__(
        self,
        root,
        burst_size=8,
        crop_sz=80,
        center_crop=False,
        random_flip=False,
        split="train",
    ):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        assert burst_size <= 14, "burst_sz must be less than or equal to 14"
        assert crop_sz <= 80, "crop_sz must be less than or equal to 80"
        assert split in ["train", "val"]

        root = root + "/" + split
        super().__init__()

        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.root = root

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = field(default_factory=self._get_burst_list)

    def _get_burst_list(self):
        burst_list = sorted(os.listdir("{}".format(self.root)))
        # print(burst_list)
        return burst_list

    def get_burst_info(self, burst_id):
        burst_info = {"burst_size": 14, "burst_name": self.burst_list[burst_id]}
        return burst_info

    def _get_raw_image(self, burst_id, im_id):
        raw_image = SamsungImage.load(
            "{}/{}/samsung_{:02d}".format(self.root, self.burst_list[burst_id], im_id)
        )
        return raw_image

    def _get_gt_image(self, burst_id):
        canon_im = CanonImage.load(
            "{}/{}/canon".format(self.root, self.burst_list[burst_id]), split=self.split
        )
        return canon_im

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        gt = self._get_gt_image(burst_id)
        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info

    def _sample_images(self):
        burst_size = 14

        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [
            0,
        ] + ids
        return ids

    def __len__(self):
        return len(self.burst_list)

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        # Read the burst images along with HR ground truth
        frames, gt, meta_info = self.get_burst(index, im_ids)

        # Extract crop if needed
        if frames[0].shape()[-1] != self.crop_sz:
            if getattr(self, "center_crop", False):
                r1 = (frames[0].shape()[-2] - self.crop_sz) // 2
                c1 = (frames[0].shape()[-1] - self.crop_sz) // 2
            else:
                r1 = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
                c1 = random.randint(0, frames[0].shape()[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            scale_factor = gt.shape()[-1] // frames[0].shape()[-1]
            frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

            gt = gt.get_crop(
                scale_factor * r1,
                scale_factor * r2,
                scale_factor * c1,
                scale_factor * c2,
            )

        # Load the RAW image data
        burst_image_data = [
            im.get_image_data(
                normalize=True,
                substract_black_level=self.substract_black_level,
                white_balance=self.white_balance,
            )
            for im in frames
        ]

        # Convert to tensor
        gt_image_data = gt.get_image_data(
            normalize=True,
            white_balance=self.white_balance,
            substract_black_level=self.substract_black_level,
        )

        if self.random_flip:
            burst_image_data = [flatten_raw_image(im) for im in burst_image_data]

            pad = [0, 0, 0, 0]
            if random.random() > 0.5:
                burst_image_data = [
                    im.flip(
                        [
                            1,
                        ]
                    )[:, 1:-1].contiguous()
                    for im in burst_image_data
                ]
                gt_image_data = gt_image_data.flip(
                    [
                        2,
                    ]
                )[:, :, 2:-2].contiguous()
                pad[1] = 1

            if random.random() > 0.5:
                burst_image_data = [
                    im.flip(
                        [
                            0,
                        ]
                    )[1:-1, :].contiguous()
                    for im in burst_image_data
                ]
                gt_image_data = gt_image_data.flip(
                    [
                        1,
                    ]
                )[:, 2:-2, :].contiguous()
                pad[3] = 1

            burst_image_data = [pack_raw_image(im) for im in burst_image_data]
            burst_image_data = [
                F.pad(im.unsqueeze(0), pad, mode="replicate").squeeze(0)
                for im in burst_image_data
            ]

            gt_image_data = F.pad(
                gt_image_data.unsqueeze(0), [4 * p for p in pad], mode="replicate"
            ).squeeze(0)

        burst_image_meta_info = frames[0].get_all_meta_data()

        burst_image_meta_info["black_level_subtracted"] = self.substract_black_level
        burst_image_meta_info["while_balance_applied"] = self.white_balance
        burst_image_meta_info["norm_factor"] = frames[0].norm_factor

        gt_image_meta_info = gt.get_all_meta_data()

        burst = torch.stack(burst_image_data, dim=0)

        burst_exposure = frames[0].get_exposure_time()
        canon_exposure = gt.get_exposure_time()

        burst_f_number = frames[0].get_f_number()
        canon_f_number = gt.get_f_number()

        burst_iso = frames[0].get_iso()
        canon_iso = gt.get_iso()

        # Normalize the GT image to account for differences in exposure, ISO etc
        light_factor_burst = burst_exposure * burst_iso / (burst_f_number**2)
        light_factor_canon = canon_exposure * canon_iso / (canon_f_number**2)

        exp_scale_factor = light_factor_burst / light_factor_canon
        gt_image_data = gt_image_data * exp_scale_factor

        gt_image_meta_info["black_level_subtracted"] = self.substract_black_level
        gt_image_meta_info["while_balance_applied"] = self.white_balance
        gt_image_meta_info["norm_factor"] = gt.norm_factor / exp_scale_factor

        burst_image_meta_info["exposure"] = burst_exposure
        burst_image_meta_info["f_number"] = burst_f_number
        burst_image_meta_info["iso"] = burst_iso

        gt_image_meta_info["exposure"] = canon_exposure
        gt_image_meta_info["f_number"] = canon_f_number
        gt_image_meta_info["iso"] = canon_iso

        burst = burst.float()
        frame_gt = gt_image_data.float()

        meta_info_burst = burst_image_meta_info
        meta_info_gt = gt_image_meta_info

        del meta_info_gt["crop_info"]

        for k, v in meta_info_gt.items():
            if isinstance(v, (list, tuple)):
                meta_info_gt[k] = torch.tensor(v)

        for k, v in meta_info_burst.items():
            if isinstance(v, (list, tuple)):
                meta_info_burst[k] = torch.tensor(v)

        meta_info_burst["burst_name"] = meta_info["burst_name"]

        return burst, frame_gt, meta_info_burst, meta_info_gt
