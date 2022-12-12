from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch._dynamo.config
import torch._inductor.config
from hypothesis import given
from hypothesis import strategies as st

from bsrt.data_processing.synthetic_burst_generator import (
    get_tmat,
    numpy_fused_rotate_get_tmat,
    numpy_fused_scale_rotate_and_shear_translate_get_tmat,
    numpy_fused_scale_rotate_get_tmat,
    numpy_get_tmat,
    torch_fused_get_tmat1,
    torch_get_tmat,
)

torch._dynamo.config.dynamic_shapes = False
torch._dynamo.config.print_graph_breaks = True
torch._dynamo.config.cache_size_limit = 8
torch.set_float32_matmul_precision("high")


def given_get_tmat_args(f):
    return given(
        image_shape=st.tuples(
            st.integers(min_value=32, max_value=1000),
            st.integers(min_value=32, max_value=1000),
        ),
        translation=st.tuples(
            st.floats(min_value=-8, max_value=8),
            st.floats(min_value=-8, max_value=8),
        ),
        theta=st.floats(min_value=-180.0, max_value=180.0),
        shear_values=st.tuples(
            st.floats(min_value=-1.0, max_value=1.0),
            st.floats(min_value=-1.0, max_value=1.0),
        ),
        scale_factors=st.tuples(
            st.floats(min_value=0.01, max_value=1.0),
            st.floats(min_value=0.01, max_value=1.0),
        ),
    )(f)


@given_get_tmat_args
def test_get_tmat_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert tmat.shape == (2, 3), f"{tmat.shape} != (3, 3)"


@given_get_tmat_args
def test_get_tmat_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert tmat.dtype == np.float64, f"{tmat.dtype} != np.float64"


@pytest.mark.parametrize(
    "impl",
    [
        numpy_get_tmat,
        numpy_fused_rotate_get_tmat,
        numpy_fused_scale_rotate_get_tmat,
        numpy_fused_scale_rotate_and_shear_translate_get_tmat,
        torch_get_tmat,
        torch_fused_get_tmat1,
    ],
)
@given_get_tmat_args
def test_get_tmat_shape_eq_impl_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64] | torch.Tensor],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert expected.shape == actual.shape, f"{expected.shape} != {actual.shape}"


@pytest.mark.parametrize(
    "impl",
    [
        numpy_get_tmat,
        numpy_fused_rotate_get_tmat,
        numpy_fused_scale_rotate_get_tmat,
        numpy_fused_scale_rotate_and_shear_translate_get_tmat,
        torch_get_tmat,
        torch_fused_get_tmat1,
    ],
)
@given_get_tmat_args
def test_get_tmat_dtype_eq_impl_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64] | torch.Tensor],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    if isinstance(actual, torch.Tensor):
        expected = torch.from_numpy(expected)
    assert expected.dtype == actual.dtype, f"{expected.dtype} != {actual.dtype}"


@pytest.mark.parametrize(
    "impl",
    [
        numpy_get_tmat,
        numpy_fused_rotate_get_tmat,
        numpy_fused_scale_rotate_get_tmat,
        numpy_fused_scale_rotate_and_shear_translate_get_tmat,
        torch_get_tmat,
        torch_fused_get_tmat1,
    ],
)
@given_get_tmat_args
def test_get_tmat_values_eq_impl_values(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64] | torch.Tensor],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    if isinstance(actual, torch.Tensor):
        expected = torch.from_numpy(expected).to(device=actual.device, dtype=actual.dtype)
        assert torch.allclose(expected, actual), f"{expected} != {actual}"
    else:
        assert np.allclose(expected, actual), f"{expected} != {actual}"


# TODO: FAST4 which has just one matrix
