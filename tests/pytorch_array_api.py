from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from hypothesis import strategies as st
from hypothesis.extra._array_helpers import Shape
from hypothesis.extra.array_api import DataType, make_strategies_namespace

torch.set_grad_enabled(False)

xps: ArrayNamespace = make_strategies_namespace(torch, api_version="draft")  # type: ignore


class ArrayNamespace(SimpleNamespace):
    def array_shapes(
        self,
        *,
        min_dims: int = 1,
        max_dims: Optional[int] = None,
        min_side: int = 1,
        max_side: Optional[int] = None,
    ) -> st.SearchStrategy[Shape]:
        ...

    def from_dtype(
        self,
        dtype: Union[DataType, str],
        *,
        min_value: Optional[int | float] = None,
        max_value: Optional[int | float] = None,
        allow_nan: Optional[bool] = None,
        allow_infinity: Optional[bool] = None,
        allow_subnormal: Optional[bool] = None,
        exclude_min: Optional[bool] = None,
        exclude_max: Optional[bool] = None,
    ) -> st.SearchStrategy[bool | int | float]:
        ...

    def arrays(
        self,
        dtype: DataType | str | st.SearchStrategy[DataType] | st.SearchStrategy[str],
        shape: int | Shape | st.SearchStrategy[Shape],
        *,
        elements: Optional[Mapping[str, Any] | st.SearchStrategy] = None,
        fill: Optional[st.SearchStrategy[Any]] = None,
        unique: bool = False,
    ) -> st.SearchStrategy:
        ...

    def scalar_dtypes(self) -> st.SearchStrategy[DataType]:
        ...

    def boolean_dtypes(self) -> st.SearchStrategy[DataType]:
        ...

    def real_dtypes(self) -> st.SearchStrategy[DataType]:
        ...

    def numeric_dtypes(self) -> st.SearchStrategy[DataType]:
        ...

    def integer_dtypes(
        self, *, sizes: int | Sequence[int] = (8, 16, 32, 64)
    ) -> st.SearchStrategy[DataType]:
        ...

    def unsigned_integer_dtypes(
        self, *, sizes: int | Sequence[int] = (8, 16, 32, 64)
    ) -> st.SearchStrategy[DataType]:
        ...

    def floating_dtypes(
        self, *, sizes: int | Sequence[int] = (32, 64)
    ) -> st.SearchStrategy[DataType]:
        ...

    # Override full so it conforms to the array API spec
    def full(
        self, shape: int | tuple[int, ...], fill_value: int | float, dtype: Optional[DataType]
    ) -> torch.Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        return torch.full(shape, fill_value, dtype=dtype)
