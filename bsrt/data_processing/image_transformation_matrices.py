import math

import torch
from torch import Tensor


def get_translate_mat(
    translation: tuple[float, float],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a translation matrix corresponding to the input translation parameters"""
    translate_x, translate_y = translation

    return torch.tensor(
        [[1.0, 0.0, translate_x], [0.0, 1.0, translate_y], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def get_shear_mat(
    image_shape: tuple[int, int],
    shear_values: tuple[float, float],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a shear matrix corresponding to the input shear parameters"""
    image_y, image_x = image_shape
    shear_x, shear_y = shear_values

    return torch.tensor(
        [
            [1.0, shear_x, -0.5 * shear_x * image_x],
            [shear_y, 1.0, -0.5 * shear_y * image_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def get_rotate_mat(
    image_shape: tuple[int, int],
    theta: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a rotation matrix corresponding to the input rotation angle, around the center of
    the image"""
    image_y, image_x = image_shape
    image_middle_x = image_x * 0.5
    image_middle_y = image_y * 0.5
    rad = math.radians(-theta)
    sin_rad = math.sin(rad)
    cos_rad = math.cos(rad)

    return torch.tensor(
        [
            [
                cos_rad,
                -sin_rad,
                image_middle_x - image_middle_x * cos_rad + image_middle_y * sin_rad,
            ],
            [
                sin_rad,
                cos_rad,
                image_middle_y - image_middle_x * sin_rad - image_middle_y * cos_rad,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def get_scale_mat(
    scale_factors: tuple[float, float],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a scaling matrix corresponding to the input scaling factors"""
    scale_x, scale_y = scale_factors

    return torch.tensor(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def _get_tmat_reference(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generates a transformation matrix corresponding to the input transformation parameters"""
    translate_mat = get_translate_mat(translation, dtype, device)
    shear_mat = get_shear_mat(image_shape, shear_values, dtype, device)
    rotate_mat = get_rotate_mat(image_shape, theta, dtype, device)
    scale_mat = get_scale_mat(scale_factors, dtype, device)

    expected = scale_mat.mm(rotate_mat).mm(shear_mat).mm(translate_mat)
    expected = expected[:2, :]
    return expected


def get_tmat(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generates a transformation matrix corresponding to the input transformation parameters"""
    image_y, image_x = image_shape
    image_middle_x = image_x * 0.5
    image_middle_y = image_y * 0.5
    translate_x, translate_y = translation
    shear_x, shear_y = shear_values
    scale_x, scale_y = scale_factors
    rad = math.radians(-theta)
    sin_rad = math.sin(rad)
    cos_rad = math.cos(rad)

    # Translate the image by the given translation factors. Then shears the image using the given
    # shear factors, about the center of the image.
    # See https://lectureloops.com/shear-transformation/ for more.
    # It is shear @ translate from pure_python_get_tmat.
    shear_after_translate = torch.tensor(
        [
            [1.0, shear_x, shear_x * translate_y - 0.5 * shear_x * image_x + translate_x],
            [shear_y, 1.0, shear_y * translate_x - 0.5 * shear_y * image_y + translate_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    # This matrix performs the rotation about the center of the image and scales.
    # It is equivalent to the following:
    #   1. Translate the center of the image to the origin
    #   2. Rotate the image about the origin
    #   3. Translate the center of the image back to where it was
    #   4. Scale by scale factors
    # It is scale @ unshift_center_to_origin @ rotate @ shift_center_to_origin from
    # pure_python_get_tmat.
    scale_after_rotate = torch.tensor(
        [
            [
                scale_x * cos_rad,
                -scale_x * sin_rad,
                scale_x * (image_middle_x * (1 - cos_rad) + image_middle_y * sin_rad),
            ],
            [
                scale_y * sin_rad,
                scale_y * cos_rad,
                scale_y * (image_middle_y * (1 - cos_rad) - image_middle_x * sin_rad),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    expected = scale_after_rotate.mm(shear_after_translate)
    expected = expected[:2, :]
    return expected
