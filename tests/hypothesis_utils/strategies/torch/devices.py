import torch
from torch.backends import mps
from hypothesis import settings
from hypothesis import strategies as st


if "_cpu" in settings._current_profile:
    torch_devices = st.just("cpu")
else:
    gpu_device_name: None | str = None
    if torch.cuda.is_available():
        gpu_device_name = "cuda"
    elif mps.is_available():
        gpu_device_name = "mps"

    if gpu_device_name is None:
        print("No GPU available, falling back to CPU")
        torch_devices = st.just("cpu")
    elif "_gpu" in settings._current_profile:
        torch_devices = st.just(gpu_device_name)
    else:
        torch_devices = st.sampled_from(["cpu", gpu_device_name])

torch_devices = torch_devices.map(torch.device)
