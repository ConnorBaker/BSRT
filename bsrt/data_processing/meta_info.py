from dataclasses import dataclass, field
from typing import Union

from torch import Tensor

from bsrt.data_processing.noises import Noises
from bsrt.data_processing.rgb_gains import RgbGains


@dataclass
class MetaInfo:
    rgb2cam: Tensor
    cam2rgb: Tensor
    smoothstep: bool
    compress_gamma: bool
    norm_factor: float = 1.0
    black_level_subtracted: bool = False
    black_level: Union[Tensor, None] = None
    while_balance_applied: bool = False
    cam_wb: Union[Tensor, None] = None
    gains: RgbGains = field(default_factory=lambda: RgbGains(0.0, 0.0, 0.0))
    noises: Noises = field(default_factory=lambda: Noises(0.0, 0.0))
