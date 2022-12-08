import torch
from hypothesis import strategies as st

torch_int_dtypes = st.sampled_from([torch.int8, torch.int16, torch.int32, torch.int64])
torch_float_dtypes = st.sampled_from([torch.bfloat16, torch.float16, torch.float32, torch.float64])
torch_real_dtypes = st.sampled_from([torch_int_dtypes, torch_float_dtypes]).flatmap(lambda x: x)
