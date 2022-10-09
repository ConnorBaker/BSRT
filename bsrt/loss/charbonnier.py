from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor


@dataclass(eq=False, init=False)
class CharbonnierLoss(nn.Module):
    """L1 charbonnier loss."""

    epsilon: float = 1e-3
    reduce: bool = True

    def __init__(self, epsilon: float = 1e-3, reduce: bool = True) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.reduce = reduce

    # TODO: Does this handle batched inputs in the way we want?
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        diff = X - Y
        error = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        if self.reduce:
            loss = torch.mean(error)
        else:
            loss = error
        return loss
