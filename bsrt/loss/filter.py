from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class Filter(nn.Module):
    n_colors: int
    conv: nn.Conv2d = field(init=False)
    loss: nn.L1Loss = field(init=False, default_factory=nn.L1Loss)

    def __post_init__(self) -> None:
        self.conv = nn.Conv2d(self.n_colors, self.n_colors, 3, 3)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype=torch.float32)
            )
        super().__init__()

    def forward(self, x, y):
        preds_x = self.conv(x)
        preds_y = self.conv(y)

        return self.loss(preds_x, preds_y)
