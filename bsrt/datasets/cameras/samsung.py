from pathlib import Path
import torch
import cv2
import numpy as np
import pickle as pkl


class SamsungImage:
    @staticmethod
    def load(path: Path):
        im_raw = cv2.imread((path / "im_raw.png").as_posix(), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open((path / "meta_info.pkl").as_posix(), "rb", -1))

        return SamsungImage(
            im_raw,
            meta_data["black_level"],
            meta_data["cam_wb"],
            meta_data["daylight_wb"],
            meta_data["color_matrix"],
            meta_data["exif_data"],
            meta_data.get("crop_info", None),
            meta_data.get("im_preview", None),
        )

    def __init__(
        self,
        im_raw,
        black_level,
        cam_wb,
        daylight_wb,
        color_matrix,
        exif_data,
        crop_info=None,
        im_preview=None,
    ):
        self.im_raw = im_raw

        self.black_level = black_level
        self.cam_wb = cam_wb
        self.daylight_wb = daylight_wb
        self.color_matrix = color_matrix
        self.exif_data = exif_data
        self.crop_info = crop_info
        self.im_preview = im_preview

        self.norm_factor = 1023.0

    def get_all_meta_data(self):
        return {
            "black_level": self.black_level,
            "cam_wb": self.cam_wb,
            "daylight_wb": self.daylight_wb,
            "color_matrix": self.color_matrix.tolist(),
        }

    def get_exposure_time(self):
        return self.exif_data["Image ExposureTime"].values[0].decimal()

    def get_noise_profile(self):
        noise = self.exif_data["Image Tag 0xC761"].values
        noise = [n[0] for n in noise]
        noise = np.array(noise).reshape(3, 2)
        return noise

    def get_f_number(self):
        return self.exif_data["Image FNumber"].values[0].decimal()

    def get_iso(self):
        return self.exif_data["Image ISOSpeedRatings"].values[0]

    def get_image_data(
        self, substract_black_level=False, white_balance=False, normalize=False
    ):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        if self.im_preview is not None:
            im_preview = self.im_preview[2 * r1 : 2 * r2, 2 * c1 : 2 * c2]
        else:
            im_preview = None

        return SamsungImage(
            im_raw,
            self.black_level,
            self.cam_wb,
            self.daylight_wb,
            self.color_matrix,
            self.exif_data,
            im_preview=im_preview,
        )

    def postprocess(self, return_np=True, norm_factor=None):
        # Convert to rgb
        # im = torch.from_numpy(self.im_raw.astype(np.float32))
        im = self.im_raw

        im = (im - torch.tensor(self.black_level).view(4, 1, 1)) * torch.tensor(
            self.cam_wb
        ).view(4, 1, 1)

        if norm_factor is None:
            im = im / im.max()
        else:
            im = im / norm_factor

        im = torch.stack((im[0], (im[1] + im[2]) / 2, im[3]), dim=0)
        # im = torch.stack((im[0], im[1], im[3]), dim=0)

        im_out = im.clamp(0.0, 1.0)

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out
