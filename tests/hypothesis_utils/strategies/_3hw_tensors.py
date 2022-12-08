from functools import partial

import torch
from hypothesis import settings
from hypothesis import strategies as st

from tests.hypothesis_utils.strategies.chw_tensors import CHWShape, chw_tensors

_3HW_TENSORS = partial(
    chw_tensors,
    shape=st.tuples(st.just(3), st.integers(32, 256), st.integers(32, 256))
    # Ensure all images have a height and width divisible by four
    .filter(lambda t: t[1] % 4 == 0 and t[2] % 4 == 0).map(CHWShape),
    # torch.float16 is excluded because it has low accuracy and some operations don't support it
    dtype=st.sampled_from([torch.bfloat16, torch.float32, torch.float64]),
)

if settings._current_profile == "ci":
    _3HW_TENSORS = partial(
        _3HW_TENSORS,
        shape=st.tuples(st.just(3), st.integers(32, 1024), st.integers(32, 1024))
        # Ensure all images have a height and width divisible by four
        .filter(lambda t: t[1] % 4 == 0 and t[2] % 4 == 0).map(CHWShape),
    )
