from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch._dynamo.config
import torch._inductor.config
from hypothesis import given
from hypothesis import strategies as st

from bsrt.data_processing.image_transformation_matrices import (
    _get_tmat_reference as torch_get_tmat_reference,
)
from bsrt.data_processing.image_transformation_matrices import get_tmat as torch_get_tmat
from bsrt.data_processing.synthetic_burst_generator import get_tmat


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
    expected = (2, 3)
    actual = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    expected = np.float64
    actual = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).dtype
    assert expected == actual


@pytest.mark.parametrize(
    "impl",
    [
        torch_get_tmat_reference,
        torch_get_tmat,
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
    ).shape
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    assert expected == actual


@pytest.mark.parametrize(
    "impl",
    [
        torch_get_tmat_reference,
        torch_get_tmat,
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
    expected_tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual_tmat = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )

    if isinstance(actual_tmat, torch.Tensor):
        expected = torch.from_numpy(expected_tmat).dtype
    else:
        expected = expected_tmat.dtype

    actual = actual_tmat.dtype

    assert expected == actual


@pytest.mark.parametrize(
    "impl",
    [
        torch_get_tmat_reference,
        torch_get_tmat,
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
