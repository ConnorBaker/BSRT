import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bsrt.model import arch_util
from bsrt.utils.bilinear_upsample_2d import bilinear_upsample_2d


@dataclass(eq=False)
class BasicModule(nn.Module):
    """Basic Module for SpyNet."""

    channel_steps: ClassVar[Sequence[int]] = [8, 32, 64, 32, 16, 2]

    net: nn.Sequential = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # We drop the first element because we do not want to apply a ReLU to the input.
            *[
                layer
                for in_channels, out_channels in zip(
                    BasicModule.channel_steps[:-1], BasicModule.channel_steps[1:]
                )
                for layer in [
                    nn.LeakyReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                ]
            ][1:]
        )

    def forward(self, tensor_input: Tensor) -> Tensor:
        return self.net(tensor_input)


@dataclass(eq=False)
class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    weights_url: ClassVar[
        str
    ] = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth"  # noqa: E501

    return_levels: Sequence[int] = field(default_factory=lambda: [5])
    net: nn.ModuleList = field(init=False)
    mean: Tensor = field(init=False)
    std: Tensor = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.net = nn.ModuleList([BasicModule() for _ in range(6)])

        weights_dict = torch.hub.load_state_dict_from_url(SpyNet.weights_url)
        self.load_state_dict(weights_dict["params"])

        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input: Tensor) -> Tensor:
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(
        self, ref: Tensor, supp: Tensor, w: int, h: int, w_floor: int, h_floor: int
    ) -> List[Tensor]:
        flow_list: List[Tensor] = []

        _ref: List[Tensor] = [self.preprocess(ref)]
        _supp: List[Tensor] = [self.preprocess(supp)]

        # FIXME: By repeatedly averaging these values, we can end up with a tensor where the width
        # and height are one.
        for _ in range(5):
            _ref.insert(
                0,
                F.avg_pool2d(input=_ref[0], kernel_size=2, stride=2, count_include_pad=False),
            )
            _supp.insert(
                0,
                F.avg_pool2d(input=_supp[0], kernel_size=2, stride=2, count_include_pad=False),
            )

        flow: Tensor = _ref[0].new_zeros(
            [
                _ref[0].size(0),
                2,
                # FIXME: Continued: taking the floor of these values can yield a tensor of width
                # or height zero.
                _ref[0].size(2) // 2,
                _ref[0].size(3) // 2,
            ]
        )

        for level in range(len(_ref)):
            # FIXME: Continued: we cannot upsample a tensor with width or height zero.
            upsampled_flow = bilinear_upsample_2d(flow, scale_factor=2, align_corners=True) * 2.0

            if upsampled_flow.size(2) != _ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode="replicate")
            if upsampled_flow.size(3) != _ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode="replicate")

            flow = (
                self.net[level](
                    torch.cat(
                        [
                            _ref[level],
                            arch_util.flow_warp(
                                _supp[level],
                                upsampled_flow.permute(0, 2, 3, 1),
                                padding_mode="border",
                            ),
                            upsampled_flow,
                        ],
                        1,
                    )
                )
                + upsampled_flow
            )

            if level in self.return_levels:
                scale: int = 2 ** (
                    5 - level
                )  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = bilinear_upsample_2d(
                    flow,
                    size=(h // scale, w // scale),
                )
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)

                flow_out.clamp(-250, 250)

                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref: Tensor, supp: Tensor) -> Union[Tensor, List[Tensor]]:
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = bilinear_upsample_2d(
            ref,
            size=(h_floor, w_floor),
        )
        supp = bilinear_upsample_2d(
            supp,
            size=(h_floor, w_floor),
        )

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list
