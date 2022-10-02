from dataclasses import dataclass, field
from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


@dataclass
class VGG(nn.Module):
    conv_index: str = "22"
    rgb_range: int = 1
    vgg: nn.Sequential = field(init=False)
    sub_mean: common.MeanShift = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        modules = list(models.vgg19(pretrained=True).features.modules())
        if self.conv_index.find("22") >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif self.conv_index.find("54") >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (
            0.229 * self.rgb_range,
            0.224 * self.rgb_range,
            0.225 * self.rgb_range,
        )
        self.sub_mean = common.MeanShift(self.rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    # TODO: Why build sub_mean if we're not going to use it?
    def forward(self, sr, hr):
        def _forward(x):
            # x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        sr = sr.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
