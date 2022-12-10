import torch
from hypothesis import strategies as st

torch_memory_formats: st.SearchStrategy[torch.memory_format] = st.sampled_from(
    [torch.contiguous_format, torch.channels_last]
)
