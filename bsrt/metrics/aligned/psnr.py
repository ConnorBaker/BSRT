from __future__ import annotations
from dataclasses import dataclass, field
from metrics.l2 import L2
from metrics.psnr import PSNR
from metrics.aligned.l2 import AlignedL2
from typing import ClassVar
import torch


@dataclass
class AlignedPSNR(PSNR):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    sr_factor: int = 4
    boundary_ignore: int | None = None
    max_value: float = 1.0
    l2: L2 = field(init=False)

    def __post_init__(self) -> None:
        super().__init__(boundary_ignore=self.boundary_ignore)
        self.l2 = AlignedL2(
            alignment_net=self.alignment_net,
            sr_factor=self.sr_factor,
            boundary_ignore=self.boundary_ignore,
        )
