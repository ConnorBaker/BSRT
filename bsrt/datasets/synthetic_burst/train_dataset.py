from dataclasses import dataclass, field
from typing import TypedDict

import torch
from torch import Tensor
from torchvision.transforms import CenterCrop, ConvertImageDtype, RandomCrop

from bsrt.data_processing.synthetic_burst_generation import (
    ImageProcessingParams,
    ImageTransformationParams,
    rgb2rawburst,
)
from bsrt.utils.types import InterpolationType


class TrainData(TypedDict):
    """
    burst: Generated LR RAW burst, a torch tensor of shape
    [burst_size, 4, self.crop_sz / (2*self.downsample_factor), self.crop_sz / (2*self.downsample_factor)]
    The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
    The extra factor 2 in the denominator (2*self.downsample_factor) corresponds to the mosaicking
    operation.

    frame_gt: The HR RGB ground truth in the linear sensor space, a torch tensor of shape
                [3, self.crop_sz, self.crop_sz]

    flow_vectors: The ground truth flow vectors between a burst image and the base image (i.e. the first image in the burst).
    The flow_vectors can be used to warp the burst images to the base frame, using the 'warp'
    function in utils.warp package.
    flow_vectors is torch tensor of shape
    [burst_size, 2, self.crop_sz / self.downsample_factor, self.crop_sz / self.downsample_factor].
    Note that the flow_vectors are in the LR RGB space, before mosaicking. Hence it has twice
    the number of rows and columns, compared to the output burst.

    NOTE: The flow_vectors are only available during training for the purpose of using any auxiliary losses if needed. The flow_vectors will NOT be provided for the bursts in the test set

    meta_info: A dictionary containing the parameters used to generate the synthetic burst.
    """

    burst: Tensor
    gt: Tensor
    flow_vectors: Tensor
    # meta_info: MetaInfo


@dataclass
class TrainDataProcessor:
    """Synthetic burst dataset for joint denoising, demosaicking, and super-resolution. RAW Burst sequences are
    synthetically generated on the fly as follows. First, a single image is loaded from the base_dataset. The sampled
    image is converted to linear sensor space using the inverse camera pipeline employed in [1]. A burst
    sequence is then generated by adding random translations and rotations to the converted image. The generated burst
    is then converted is then mosaicked, and corrupted by random noise to obtain the RAW burst.

    [1] Unprocessing Images for Learned Raw Denoising, Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen,
    Jiawen and Sharlet, Dillon and Barron, Jonathan T, CVPR 2019
    """

    burst_size: int
    crop_sz: int
    dtype: torch.dtype
    final_crop_sz: int = field(init=False)
    downsample_factor: int = 4
    burst_transformation_params: ImageTransformationParams = ImageTransformationParams(
        max_translation=24.0,
        max_rotation=1.0,
        max_shear=0.0,
        max_scale=0.0,
        border_crop=24,
    )
    image_processing_params: ImageProcessingParams = ImageProcessingParams(
        random_ccm=False,
        random_gains=False,
        smoothstep=False,
        compress_gamma=False,
        add_noise=False,
    )
    interpolation_type: InterpolationType = "bilinear"

    def __post_init__(self):
        self.final_crop_sz = self.crop_sz + 2 * self.burst_transformation_params.border_crop
        self.cropper = RandomCrop(self.final_crop_sz)
        self.boundary_ignorer = CenterCrop(
            self.crop_sz - self.burst_transformation_params.border_crop
        )
        self.pre_dtype_converter = ConvertImageDtype(torch.float32)
        self.post_dtype_converter = ConvertImageDtype(self.dtype)

    def __call__(self, frame: Tensor) -> TrainData:
        # Extract a random crop from the image
        cropped_frame: Tensor = self.cropper(frame)
        converted_frame: Tensor = self.pre_dtype_converter(cropped_frame)

        burst, gt, _burst_rgb, flow_vectors, meta_info = rgb2rawburst(
            converted_frame,
            self.burst_size,
            self.downsample_factor,
            burst_transformation_params=self.burst_transformation_params,
            image_processing_params=self.image_processing_params,
            interpolation_type=self.interpolation_type,
        )
        burst = self.post_dtype_converter(burst)
        gt = self.post_dtype_converter(self.boundary_ignorer(gt))
        flow_vectors = self.post_dtype_converter(flow_vectors)

        return TrainData(
            burst=burst,
            gt=gt,
            flow_vectors=flow_vectors,
        )
