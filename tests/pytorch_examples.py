from copy import copy
from typing import ClassVar, Literal, Tuple, Type, Union, TypeVar

import torch
import torch._C
from torch import Tensor
from numpy.lib.mixins import NDArrayOperatorsMixin

from crosshair import (
    IgnoreAttempt,
    SymbolicFactory,
    deep_realize,
    realize,
    register_type,
    register_patch,
)

#
# Classes implemented in C generally cannot be simulated symbolically by
# CrossHair.
# However, you can install hooks to produce custom symbolic values.
# Here, we provide a lazy version of the numpy's array class: this has
# a symbolic shape and data type.
# When an actual operation needs to be performed, we'll construct the
# actual array.
#


# class SymbolicDType:
#     float_dtypes: ClassVar[list[torch.dtype]] = [
#         torch.float32,
#         torch.float,
#         torch.float64,
#         torch.double,
#         torch.float16,
#         torch.bfloat16,
#     ]
#     num_float_dtypes: ClassVar[int] = len(float_dtypes)

#     def __init__(self, creator: SymbolicFactory):
#         self.idx = creator(int, "_idx_dtype")

#     def __ch_realize__(self):
#         concrete_idx = realize(self.idx)
#         if 0 <= concrete_idx < SymbolicDType.num_float_dtypes:
#             return SymbolicDType.float_dtypes[concrete_idx]
#         else:
#             raise IgnoreAttempt("Unknown dtype")


# register_type(torch.dtype, SymbolicDType)


class SymbolicTensor:
    def __init__(self, creator: SymbolicFactory):
        self.shape = creator(tuple[int, ...], "_shape")
        self.ndims = self.shape.__len__()
        # self.dtype = creator(torch.dtype, "_dtype")
        self.dtype = torch.float32

    def __deepcopy__(self, memo):
        c = object.__new__(type(self))
        c.shape = copy(self.shape)
        c.ndims = copy(self.ndims)
        c.dtype = copy(self.dtype)
        return c

    def __getattr__(self, name):
        return getattr(self.__ch_realize__(), name)

    def __ch_realize__(self):
        concrete_dtype = realize(self.dtype)
        concrete_shape = realize(self.shape)
        if any(realize(size) < 0 for size in concrete_shape):
            raise IgnoreAttempt("pytorch disallows negative dimensions")
        t = torch.rand(concrete_shape, dtype=concrete_dtype)
        return t


# Make crosshair use our custom class whenever it needs a symbolic
register_type(Tensor, SymbolicTensor)


def matrix_multiply(image1: Tensor, image2: Tensor) -> Tensor:
    """
    pre: image1.dtype == image2.dtype
    pre: image1.ndims == image2.ndims == 2
    pre: image1.shape[1] == image2.shape[0]
    post: _.shape == (image1.shape[0], image2.shape[1])
    """
    return image1 @ image2


# def threshold_image(image: Tensor, threshold: float) -> Tensor:
#     """
#     >>> threshold_image(torch.tensor([[0.0, 0.3], [0.6, 1.0]], dtype=torch.float64), 0.5)
#     array([[0.5, 0.5],
#            [0.6, 1. ]])

#     pre: len(image.shape) == 2
#     pre: image.dtype == torch.float64
#     pre: image.size > 0
#     pre: threshold > 0
#     post: _.shape == image.shape
#     post: image.dtype == _.dtype
#     post: torch.min(_) >= threshold
#     """
#     return torch.where(image > threshold, image, threshold)


# def repeat_array(src: Tensor, count: int) -> Tensor:
#     """
#     pre: src.shape[0] > 0
#     pre: count > 0
#     post: _.shape == (src.shape[0] * count, *src.shape[1:])
#     """
#     return torch.concatenate([src] * count)

# def add_commutes(a: Tensor, b: Tensor) -> bool:
#     """
#     pre: a.shape == b.shape
#     post: torch.allclose(a + b, b + a)
#     """
#     return True


# def sub_commutes(a: Tensor, b: Tensor) -> bool:
#     """
#     pre: a.shape == b.shape
#     post: torch.allclose(a - b, b - a)
#     """
#     return True
