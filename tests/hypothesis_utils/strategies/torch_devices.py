import torch
from hypothesis import settings
from hypothesis import strategies as st

if "_cpu" in settings._current_profile:
    torch_devices = st.just("cpu")
else:
    # In both the profiles with a _gpu suffix and the default profile, we want to use the GPU.
    if not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        torch_devices = st.just("cpu")
    elif "_gpu" in settings._current_profile:
        torch_devices = st.just("cuda")
    else:
        torch_devices = st.sampled_from(["cpu", "cuda"])

torch_devices = torch_devices.map(torch.device)
