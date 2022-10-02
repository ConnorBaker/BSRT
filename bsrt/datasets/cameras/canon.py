from __future__ import annotations
from pathlib import Path
import torch
import cv2
import numpy as np
import pickle as pkl
from utils.bilinear_upsample_2d import bilinear_upsample_2d


class CanonImage:
    @staticmethod
    def load(path: Path) -> CanonImage:
        im_raw = cv2.imread((path / "im_raw.png").as_posix(), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open((path / "meta_info.pkl").as_posix(), "rb", -1))

        return CanonImage(
            im_raw.float(),
            meta_data["black_level"],
            meta_data["cam_wb"],
            meta_data["daylight_wb"],
            meta_data["rgb_xyz_matrix"],
            meta_data.get("exif_data", None),
            meta_data.get("crop_info", None),
        )

    def __init__(
        self,
        im_raw,
        black_level,
        cam_wb,
        daylight_wb,
        rgb_xyz_matrix,
        exif_data,
        crop_info=None,
    ):
        super(CanonImage, self).__init__()
        self.im_raw = im_raw

        if len(black_level) == 4:
            black_level = [black_level[0], black_level[1], black_level[3]]
        self.black_level = black_level

        if len(cam_wb) == 4:
            cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
        self.cam_wb = cam_wb

        if len(daylight_wb) == 4:
            daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
        self.daylight_wb = daylight_wb

        self.rgb_xyz_matrix = rgb_xyz_matrix
        self.xyz_srgb_matrix = torch.tensor(
            [
                3.2404542,
                -1.5371385,
                -0.4985314,
                -0.9692660,
                1.8760108,
                0.0415560,
                0.0556434,
                -0.2040259,
                1.0572252,
            ]
        ).view(3, 3)
        self.exif_data = exif_data
        self.crop_info = crop_info

        self.norm_factor = 16383

    def shape(self):
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def get_all_meta_data(self):
        return {
            "black_level": self.black_level,
            "cam_wb": self.cam_wb,
            "daylight_wb": self.daylight_wb,
            "rgb_xyz_matrix": self.rgb_xyz_matrix.tolist(),
            "crop_info": self.crop_info,
            "norm_factor": self.norm_factor,
        }

    def get_exposure_time(self):
        return self.exif_data["EXIF ExposureTime"].values[0].decimal()

    def get_f_number(self):
        return self.exif_data["EXIF FNumber"].values[0].decimal()

    def get_iso(self):
        return self.exif_data["EXIF ISOSpeedRatings"].values[0]

    def get_image_data(
        self, substract_black_level=False, white_balance=False, normalize=False
    ):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def set_image_data(self, im_data):
        self.im_raw = im_data

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return CanonImage(
            im_raw,
            self.black_level,
            self.cam_wb,
            self.daylight_wb,
            self.rgb_xyz_matrix,
            self.exif_data,
            self.crop_info,
        )

    def set_crop_info(self, crop_info):
        self.crop_info = crop_info

    def resize(self, size=None, scale_factor=None):

        self.im_raw = bilinear_upsample_2d(
            self.im_raw.unsqueeze(0),
            size=size,
            scale_factor=scale_factor,
        ).squeeze(0)

    def postprocess(self, return_np=True):
        # Convert to rgb
        im = self.im_raw

        im = (im - torch.tensor(self.black_level).view(3, 1, 1)).float() * torch.tensor(
            self.cam_wb
        ).view(3, 1, 1)

        im_out = im / im.max()
        im_out = im_out.clamp(0.0, 1.0)

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out
