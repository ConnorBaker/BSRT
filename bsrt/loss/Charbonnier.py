import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """L1 charbonnier loss."""

    def __init__(self, epsilon=1e-3, reduce=True):
        super(CharbonnierLoss, self).__init__()
        self.eps = epsilon * epsilon
        self.reduce = reduce

    def forward(self, X, Y):
        # diff = X - Y
        diff = torch.add(X, -Y)
        # error = sqrt(diff * diff + eps)
        # error = sqrt(diff * diff + epsilon^2)
        # = sqrt(diff^2 + epsilon^2)
        # = sqrt((X-Y)^2 + epsilon^2))
        error = torch.sqrt(diff * diff + self.eps)
        if self.reduce:
            loss = torch.mean(error)
        else:
            loss = error
        return loss
