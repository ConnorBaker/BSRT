from dataclasses import dataclass, field
from typing import Any, Union

from torch import Tensor


@dataclass
class ImageMetadata:
    black_level: Union[list[float], None] = None
    cam_wb: Union[list[float], None] = None
    daylight_wb: Union[list[float], None] = None
    exif_data: Union[dict[str, Any], None] = None
    im_preview: Union[Tensor, None] = None
    norm_factor: float = field(init=False)
    xyz_srgb_matrix: Union[Tensor, None] = field(init=False)
