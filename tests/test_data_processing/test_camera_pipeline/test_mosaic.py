import torch
from torch import Tensor
from hypothesis import given
from tests.hypothesis_utils.strategies._3hw_tensors import _3HW_TENSORS
from bsrt.data_processing.camera_pipeline import mosaic, demosaic


# Property-based tests which ensure:
# - The mosaiced image has four channels
# - The mosaiced image is half the height of the original image
# - The mosaiced image is half the width of the original image
# - The mosaiced image has the same dtype as the original image
# - The mosaiced image is on the same device as the original image
# - The demosaiced image has the same shape as the original image
# - The demosaiced image has the same dtype as the original image
# - The demosaiced image is on the same device as the original image
# - demosaic is roughly the inverse of mosaic


@given(image=_3HW_TENSORS())
def test_mosaic_has_four_channels(image: Tensor) -> None:
    """
    Tests that the mosaiced image has four channels.

    Args:
        image: A 3HW image of floating dtype
    """
    assert mosaic(image).shape[0] == 4


@given(image=_3HW_TENSORS())
def test_mosaic_has_half_height(image: Tensor) -> None:
    """
    Tests that the mosaiced image is half the height of the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert mosaic(image).shape[1] == image.shape[1] // 2


@given(image=_3HW_TENSORS())
def test_mosaic_has_half_width(image: Tensor) -> None:
    """
    Tests that the mosaiced image is half the width of the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert mosaic(image).shape[2] == image.shape[2] // 2


@given(image=_3HW_TENSORS())
def test_mosaic_dtype_invariance(image: Tensor) -> None:
    """
    Tests that the mosaiced image has the same dtype as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert mosaic(image).dtype == image.dtype


@given(image=_3HW_TENSORS())
def test_mosaic_device_invariance(image: Tensor) -> None:
    """
    Tests that the mosaiced image is on the same device as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert mosaic(image).device == image.device


@given(image=_3HW_TENSORS())
def test_demosaic_shape_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image has the same shape as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert demosaic(mosaic(image)).shape == image.shape


@given(image=_3HW_TENSORS())
def test_demosaic_dtype_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image has the same dtype as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert demosaic(mosaic(image)).dtype == image.dtype


@given(image=_3HW_TENSORS())
def test_demosaic_device_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image is on the same device as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    assert demosaic(mosaic(image)).device == image.device


@given(image=_3HW_TENSORS())
def test_demosaic_is_inverse_of_mosaic(image: Tensor) -> None:
    """
    Tests that demosaic is roughly the inverse of mosaic.

    Args:
        image: A 3HW image of floating dtype
    """
    assert torch.allclose(demosaic(mosaic(image)), image, rtol=1e-3, atol=1e-3)
