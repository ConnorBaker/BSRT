import torch
import torch.nn as nn
import torch.nn.functional as F

from bsrt.option import Config

class HistEntropy(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, x):
        p = torch.softmax(x, dim=1)
        logp = torch.log_softmax(x, dim=1)

        entropy = (-p * logp).sum(dim=(2, 3)).mean()

        return entropy
