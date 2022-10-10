from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn


@dataclass(eq=False, init=False)
class Filter(nn.Module):
    n_colors: int
    conv: nn.Conv2d = field(init=False)
    loss: nn.L1Loss = field(init=False)

    def __init__(self, n_colors: int) -> None:
        super().__init__()
        self.n_colors = n_colors
        self.conv = nn.Conv2d(self.n_colors, self.n_colors, 3, 3)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype=torch.float32)
            )

        self.loss = nn.L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        preds_x = self.conv(x)
        preds_y = self.conv(y)

        return self.loss(preds_x, preds_y)
