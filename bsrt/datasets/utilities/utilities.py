import torch
import cv2
import numpy as np


def pack_raw_image(im_raw):
    if isinstance(im_raw, np.ndarray):
        im_out = np.zeros_like(
            im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2)
        )
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros(
            (4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype
        )
    else:
        raise Exception

    im_out[0, :, :] = im_raw[0::2, 0::2]
    im_out[1, :, :] = im_raw[0::2, 1::2]
    im_out[2, :, :] = im_raw[1::2, 0::2]
    im_out[3, :, :] = im_raw[1::2, 1::2]
    return im_out


def flatten_raw_image(im_raw_4ch):
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(
            im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2)
        )
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros(
            (im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype
        )
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]

    return im_out


def pack_raw_image_batch(im_raw):
    im_out = torch.zeros(
        (
            im_raw.shape[0],
            im_raw.shape[1],
            4,
            im_raw.shape[3] // 2,
            im_raw.shape[4] // 2,
        ),
        dtype=im_raw.dtype,
    )
    im_out[:, :, 0, :, :] = im_raw[:, :, 0, 0::2, 0::2]
    im_out[:, :, 1, :, :] = im_raw[:, :, 0, 0::2, 1::2]
    im_out[:, :, 2, :, :] = im_raw[:, :, 0, 1::2, 0::2]
    im_out[:, :, 3, :, :] = im_raw[:, :, 0, 1::2, 1::2]
    return im_out


def flatten_raw_image_batch(im_raw_4ch):
    im_out = torch.zeros(
        (
            im_raw_4ch.shape[0],
            im_raw_4ch.shape[1],
            1,
            im_raw_4ch.shape[3] * 2,
            im_raw_4ch.shape[4] * 2,
        ),
        dtype=im_raw_4ch.dtype,
    )
    im_out[:, :, 0, 0::2, 0::2] = im_raw_4ch[:, :, 0, :, :]
    im_out[:, :, 0, 0::2, 1::2] = im_raw_4ch[:, :, 1, :, :]
    im_out[:, :, 0, 1::2, 0::2] = im_raw_4ch[:, :, 2, :, :]
    im_out[:, :, 0, 1::2, 1::2] = im_raw_4ch[:, :, 3, :, :]

    return im_out
