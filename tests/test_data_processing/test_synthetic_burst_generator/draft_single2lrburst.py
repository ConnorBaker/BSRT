from typing import get_args

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from bsrt.data_processing.image_transformation_params import ImageTransformationParams
from bsrt.utils.types import InterpolationType
from tests.hypothesis_utils.strategies.torch._3hw_tensors import _3HW_TENSORS


@given(
    image=_3HW_TENSORS(dtype=torch.float32),
    burst_size=st.integers(min_value=1, max_value=20),
    downsample_factor=st.floats(0.25, 1.0),
    transformation_params=st.none(),  # TODO: Add transformation params
    interpolation_type=st.sampled_from(get_args(InterpolationType)),
)
def test_single2lrburst(
    image: Tensor,
    burst_size: int,
    downsample_factor: float,
    transformation_params: None | ImageTransformationParams,
    interpolation_type: InterpolationType,
) -> None:
    assert False, "TODO: Finish writing test"
