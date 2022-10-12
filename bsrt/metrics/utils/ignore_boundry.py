from typing import Union

from torch import Tensor
from typing_extensions import overload


@overload
def ignore_boundary(x: None, boundary_ignore: Union[int, None]) -> None:
    ...


@overload
def ignore_boundary(x: Tensor, boundary_ignore: Union[int, None]) -> Tensor:
    ...


def ignore_boundary(
    x: Union[Tensor, None], boundary_ignore: Union[int, None] = None
) -> Union[Tensor, None]:
    if boundary_ignore is not None and x is not None:
        x = x[
            ...,
            boundary_ignore:-boundary_ignore,
            boundary_ignore:-boundary_ignore,
        ]

    return x
