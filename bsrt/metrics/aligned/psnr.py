from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Union

import torch
from metrics.aligned.l2 import AlignedL2
from metrics.l2 import L2
from metrics.psnr import PSNR


# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False, init=False)
class AlignedPSNR(PSNR):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    boundary_ignore: Union[int, None] = None
    max_value: float = 1.0
    sr_factor: int = 4
    l2: L2 = field(init=False)

    # TODO: We cannot use the generated init with nn.Modules arguments because we must call the super init before we can call the module init.
    def __init__(
        self,
        alignment_net: torch.nn.Module,
        boundary_ignore: Union[int, None] = None,
        max_value: float = 1.0,
        sr_factor: int = 4,
    ) -> None:
        super().__init__()
        self.alignment_net = alignment_net
        self.boundary_ignore = boundary_ignore
        self.max_value = max_value
        self.sr_factor = sr_factor
        self.l2 = AlignedL2(
            alignment_net=self.alignment_net,
            sr_factor=self.sr_factor,
            boundary_ignore=self.boundary_ignore,
        )
