from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn


@dataclass(eq=False, init=False)
class HistEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        p = torch.softmax(x, dim=1)
        logp = torch.log_softmax(x, dim=1)
        entropy = (-p * logp).sum(dim=(2, 3)).mean()
        return entropy
