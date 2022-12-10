import torch
from hypothesis import strategies as st

torch_bool_dtypes: st.SearchStrategy[torch.dtype] = st.just(torch.bool)
torch_int_dtypes: st.SearchStrategy[torch.dtype] = st.sampled_from(
    [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
torch_float_dtypes: st.SearchStrategy[torch.dtype] = st.sampled_from(
    [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
torch_real_dtypes: st.SearchStrategy[torch.dtype] = st.sampled_from(
    [torch_int_dtypes, torch_float_dtypes]
).flatmap(lambda x: x)
torch_complex_dtypes: st.SearchStrategy[torch.dtype] = st.sampled_from(
    [torch.complex32, torch.complex64, torch.complex128]
)
