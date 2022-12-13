from functools import partial

from hypothesis import settings
from hypothesis import strategies as st

from tests.hypothesis_utils.strategies.torch.dtypes import torch_float_dtypes
from tests.hypothesis_utils.strategies.torch.nchw_tensors import NCHWShape, nchw_tensors

if settings._current_profile == "ci":
    shape_strat = st.tuples(
        st.integers(1, 128), st.just(3), st.integers(32, 1024), st.integers(32, 1024)
    )
else:
    shape_strat = st.tuples(
        st.integers(1, 16), st.just(3), st.integers(32, 256), st.integers(32, 256)
    )

_N3HW_TENSORS = partial(
    nchw_tensors,
    shape=shape_strat
    # Ensure all images have a height and width divisible by four
    .filter(lambda t: t[-2] % 4 == 0 and t[-1] % 4 == 0).map(NCHWShape),
    # torch.float16 is excluded because it has low accuracy and some operations don't support it
    # dtype=st.sampled_from([torch.bfloat16, torch.float32, torch.float64]),
    dtype=torch_float_dtypes,
)
