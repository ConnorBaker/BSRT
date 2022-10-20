from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class ImageMetadata:
    black_level: list[float] | None = None
    cam_wb: list[float] | None = None
    daylight_wb: list[float] | None = None
    exif_data: dict[str, Any] | None = None
    im_preview: Tensor | None = None
    norm_factor: float = field(init=False)
    xyz_srgb_matrix: Tensor | None = field(init=False)
