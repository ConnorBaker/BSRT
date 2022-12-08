from hypothesis import given
from torch import Tensor

from bsrt.data_processing.camera_pipeline import demosaic, mosaic
from tests.hypothesis_utils.strategies._3hw_tensors import _3HW_TENSORS

# Property-based tests which ensure:
# - The mosaiced image has four channels
# - The mosaiced image is half the height of the original image
# - The mosaiced image is half the width of the original image
# - The mosaiced image has the same dtype as the original image
# - The mosaiced image is on the same device as the original image
# - The demosaiced image has the same shape as the original image
# - The demosaiced image has the same dtype as the original image
# - The demosaiced image is on the same device as the original image


@given(image=_3HW_TENSORS())
def test_mosaic_has_four_channels(image: Tensor) -> None:
    """
    Tests that the mosaiced image has four channels.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    assert mosaiced_image.shape[0] == 4, f"Expected 4 channels, got {mosaiced_image.shape[0]}"


@given(image=_3HW_TENSORS())
def test_mosaic_has_half_height(image: Tensor) -> None:
    """
    Tests that the mosaiced image is half the height of the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    assert (
        mosaiced_image.shape[1] == image.shape[1] // 2
    ), f"Expected {image.shape[1] // 2} height, got {mosaiced_image.shape[1]}"


@given(image=_3HW_TENSORS())
def test_mosaic_has_half_width(image: Tensor) -> None:
    """
    Tests that the mosaiced image is half the width of the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    assert (
        mosaiced_image.shape[2] == image.shape[2] // 2
    ), f"Expected {image.shape[2] // 2} width, got {mosaiced_image.shape[2]}"


@given(image=_3HW_TENSORS())
def test_mosaic_dtype_invariance(image: Tensor) -> None:
    """
    Tests that the mosaiced image has the same dtype as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    assert (
        mosaiced_image.dtype == image.dtype
    ), f"Expected {image.dtype} dtype, got {mosaiced_image.dtype}"


@given(image=_3HW_TENSORS())
def test_mosaic_device_invariance(image: Tensor) -> None:
    """
    Tests that the mosaiced image is on the same device as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    assert (
        mosaiced_image.device == image.device
    ), f"Expected {image.device} device, got {mosaiced_image.device}"


@given(image=_3HW_TENSORS())
def test_demosaic_shape_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image has the same shape as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    demosaiced_image = demosaic(mosaiced_image)
    assert (
        demosaiced_image.shape == image.shape
    ), f"Expected {image.shape} shape, got {demosaiced_image.shape}"


@given(image=_3HW_TENSORS())
def test_demosaic_dtype_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image has the same dtype as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    demosaiced_image = demosaic(mosaiced_image)
    assert (
        demosaiced_image.dtype == image.dtype
    ), f"Expected {image.dtype} dtype, got {demosaiced_image.dtype}"


@given(image=_3HW_TENSORS())
def test_demosaic_device_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image is on the same device as the original image.

    Args:
        image: A 3HW image of floating dtype
    """
    mosaiced_image = mosaic(image)
    demosaiced_image = demosaic(mosaiced_image)
    assert (
        demosaiced_image.device == image.device
    ), f"Expected {image.device} device, got {demosaiced_image.device}"
