from data_processing.meta_info import MetaInfo
from data_processing.noises import Noises
from data_processing.rgb_gains import RgbGains
from data_processing.image_processing_params import ImageProcessingParams
from data_processing.image_transformation_params import ImageTransformationParams
from utils.types import InterpolationType
from data_processing.camera_pipeline import (
    random_ccm,
    apply_ccm,
    mosaic,
    invert_smoothstep,
    gamma_expansion,
)
from torch import Tensor
from utils.bilinear_upsample_2d import bilinear_upsample_2d
from utils.data_format_utils import torch_to_numpy, numpy_to_torch
import cv2
import numpy as np
import numpy.typing as npt
import random
import torch


def random_crop(
    frames: Tensor, crop_sz: float | list[float] | tuple[float, ...]
) -> Tensor:
    """Extract a random crop of size crop_sz from the input frames. If the crop_sz is larger than the input image size,
    then the largest possible crop of same aspect ratio as crop_sz will be extracted from frames, and upsampled to
    crop_sz.
    """
    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
    crop_sz_t: Tensor = torch.tensor(crop_sz).float()

    shape = frames.shape

    # Select scale_factor. Ensure the crop fits inside the image
    max_scale_factor = torch.tensor(shape[-2:]).float() / crop_sz_t
    max_scale_factor = max_scale_factor.min().item()

    if max_scale_factor < 1.0:
        scale_factor = max_scale_factor
    else:
        scale_factor = 1.0

    # Extract the crop
    orig_crop_sz = (crop_sz_t * scale_factor).floor()

    assert (
        orig_crop_sz[-2] <= shape[-2] and orig_crop_sz[-1] <= shape[-1]
    ), "Bug in crop size estimation!"

    r1 = random.randint(0, int(shape[-2] - orig_crop_sz[-2]))
    c1 = random.randint(0, int(shape[-1] - orig_crop_sz[-1]))

    r2 = r1 + orig_crop_sz[0].int().item()
    c2 = c1 + orig_crop_sz[1].int().item()

    frames_crop = frames[:, r1:r2, c1:c2]

    # Resize to crop_sz
    if scale_factor < 1.0:
        frames_crop = bilinear_upsample_2d(
            frames_crop.unsqueeze(0),
            size=tuple(crop_sz_t.int().tolist()),
        ).squeeze(0)
    return frames_crop


def rgb2rawburst(
    image: Tensor,
    burst_size: int,
    downsample_factor: float = 1,
    burst_transformation_params: ImageTransformationParams | None = None,
    image_processing_params: ImageProcessingParams | None = None,
    interpolation_type: InterpolationType = "bilinear",
) -> tuple[Tensor, Tensor, Tensor, Tensor, MetaInfo]:
    """Generates a synthetic LR RAW burst from the input image. The input sRGB image is first converted to linear
    sensor space using an inverse camera pipeline. A LR burst is then generated by applying random
    transformations defined by burst_transformation_params to the input image, and downsampling it by the
    downsample_factor. The generated burst is then mosaicekd and corrputed by random noise.
    """

    if image_processing_params is None:
        image_processing_params = ImageProcessingParams()

    # Sample camera pipeline params
    if image_processing_params.random_ccm:
        rgb2cam = random_ccm()
    else:
        rgb2cam = torch.eye(3).float()
    cam2rgb = rgb2cam.inverse()

    # Sample gains
    if image_processing_params.random_gains:
        gains = RgbGains.random_gains()
    else:
        gains = RgbGains(1.0, 1.0, 1.0)

    # Approximately inverts global tone mapping.
    if image_processing_params.smoothstep:
        image = invert_smoothstep(image)

    # Inverts gamma compression.
    if image_processing_params.compress_gamma:
        image = gamma_expansion(image)

    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)

    # Approximately inverts white balance and brightening.
    image = gains.safe_invert_gains(image)

    # Clip saturated pixels.
    image = image.clamp(0.0, 1.0)

    # Generate LR burst
    image_burst_rgb, flow_vectors = single2lrburst(
        image=image,
        burst_size=burst_size,
        downsample_factor=downsample_factor,
        transformation_params=burst_transformation_params,
        interpolation_type=interpolation_type,
    )

    # mosaic
    image_burst = mosaic(image_burst_rgb.clone())

    # Add noise
    if image_processing_params.add_noise:
        noises = Noises.random_noise_levels()
        image_burst = noises.apply(image_burst)
    else:
        noises = Noises(0.0, 0.0)

    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, 1.0)

    meta_info = MetaInfo(
        rgb2cam=rgb2cam,
        cam2rgb=cam2rgb,
        gains=gains,
        smoothstep=image_processing_params.smoothstep,
        compress_gamma=image_processing_params.compress_gamma,
        noises=noises,
    )

    return image_burst, image, image_burst_rgb, flow_vectors, meta_info


def get_tmat(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Generates a transformation matrix corresponding to the input transformation parameters"""
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array(
        [
            [1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
            [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
            [0.0, 0.0, 1.0],
        ]
    )

    t_scale = np.array(
        [[scale_factors[0], 0.0, 0.0], [0.0, scale_factors[1], 0.0], [0.0, 0.0, 1.0]]
    )

    t_mat = t_scale @ t_rot @ t_shear @ t_mat

    t_mat = t_mat[:2, :]

    return t_mat


def single2lrburst(
    image: Tensor | npt.NDArray[np.uint8],
    burst_size: int,
    downsample_factor: float = 1.0,
    transformation_params: ImageTransformationParams | None = None,
    interpolation_type: InterpolationType = "bilinear",
) -> tuple[Tensor, Tensor]:
    """Generates a burst of size burst_size from the input image by applying random transformations defined by
    transformation_params, and downsampling the resulting burst by downsample_factor.
    """
    if transformation_params is None:
        transformation_params = ImageTransformationParams()

    match interpolation_type:
        case "bilinear":
            interpolation = cv2.INTER_LINEAR
        case "lanczos":
            interpolation = cv2.INTER_LANCZOS4
        case "nearest":
            interpolation = cv2.INTER_NEAREST
        case "bicubic":
            interpolation = cv2.INTER_CUBIC

    normalize = False
    if isinstance(image, torch.Tensor):
        if image.max() < 2.0:
            image = image * 255.0
            normalize = True
        image = torch_to_numpy(image).astype(np.uint8)

    burst: list[Tensor] = []
    sample_pos_inv_all: list[Tensor] = []

    rvs, cvs = torch.meshgrid(
        [torch.arange(0, image.shape[0]), torch.arange(0, image.shape[1])],
        indexing="ij",
    )

    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()

    # For base image, do not apply any random transformations. We only translate the image to center the
    # sampling grid
    shift: float = (downsample_factor / 2.0) - 0.5
    translation: tuple[float, float] = (shift, shift)
    theta: float = 0.0
    shear_factor: tuple[float, float] = (0.0, 0.0)

    for i in range(1, burst_size):
        # Sample random image transformation parameters
        max_translation = transformation_params.max_translation

        if max_translation <= 0.01:
            shift = (downsample_factor / 2.0) - 0.5
            translation = (shift, shift)
        else:
            translation = (
                random.uniform(-max_translation, max_translation),
                random.uniform(-max_translation, max_translation),
            )

        max_rotation = transformation_params.max_rotation
        theta = random.uniform(-max_rotation, max_rotation)

        max_shear = transformation_params.max_shear
        shear_x = random.uniform(-max_shear, max_shear)
        shear_y = random.uniform(-max_shear, max_shear)
        shear_factor = (shear_x, shear_y)

        max_ar_factor = transformation_params.max_ar_factor
        ar_factor: float = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

        max_scale = transformation_params.max_scale
        scale_factor: float = np.exp(random.uniform(-max_scale, max_scale))

        scale_factor_tuple: tuple[float, float] = (
            scale_factor,
            scale_factor * ar_factor,
        )

        output_sz: tuple[int, int] = (image.shape[1], image.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        t_mat: npt.NDArray[np.float64] = get_tmat(
            (image.shape[0], image.shape[1]),
            translation,
            theta,
            shear_factor,
            scale_factor_tuple,
        )
        t_mat_tensor: Tensor = torch.from_numpy(t_mat)

        # Apply the sampled affine transformation
        image_t: npt.NDArray[np.float64] = cv2.warpAffine(
            image, t_mat, output_sz, flags=interpolation, borderMode=cv2.BORDER_CONSTANT
        )

        t_mat_tensor_3x3: Tensor = torch.cat(
            (t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0
        )
        t_mat_tensor_inverse: Tensor = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

        sample_pos_inv: Tensor = torch.mm(
            sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()
        ).view(*sample_grid.shape[:2], -1)

        if transformation_params.border_crop > 0:
            border_crop = transformation_params.border_crop

            image_t = image_t[border_crop:-border_crop, border_crop:-border_crop, :]
            sample_pos_inv = sample_pos_inv[
                border_crop:-border_crop, border_crop:-border_crop, :
            ]

        # Downsample the image
        image_t = cv2.resize(
            image_t,
            None,
            fx=1.0 / downsample_factor,
            fy=1.0 / downsample_factor,
            interpolation=interpolation,
        )
        sample_pos_inv = cv2.resize(
            sample_pos_inv.numpy(),
            None,
            fx=1.0 / downsample_factor,
            fy=1.0 / downsample_factor,
            interpolation=interpolation,
        )

        sample_pos_inv = torch.from_numpy(sample_pos_inv).permute(2, 0, 1).contiguous()

        if normalize:
            image_t_tensor = numpy_to_torch(image_t).float() / 255.0
        else:
            image_t_tensor = numpy_to_torch(image_t).float()
        burst.append(image_t_tensor)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all_tensor = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = sample_pos_inv_all_tensor - sample_pos_inv_all_tensor[:, :1, ...]

    return burst_images, flow_vectors
