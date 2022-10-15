import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.arch_util as arch_util
from utils.bilinear_upsample_2d import bilinear_upsample_2d


class BasicModule(nn.Module):
    """Basic Module for SpyNet."""

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3
            ),
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])

        weights_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth"
        )
        self.load_state_dict(weights_dict["params"])

        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        # ref = [ref]
        # supp = [supp]

        # FIXME: By repeatedly averaging these values, we can end up with a tensor where the width and height are one.
        for level in range(5):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )

        flow = ref[0].new_zeros(
            [
                ref[0].size(0),
                2,
                # FIXME: Continued: taking the floor of these values can yield a tensor of width or height zero.
                ref[0].size(2) // 2,
                ref[0].size(3) // 2,
            ]
        )

        for level in range(len(ref)):
            # FIXME: Continued: we cannot upsample a tensor with width or height zero.
            upsampled_flow = (
                bilinear_upsample_2d(flow, scale_factor=2, align_corners=True) * 2.0
            )

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 0, 0, 1], mode="replicate"
                )
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 1, 0, 0], mode="replicate"
                )

            flow = (
                self.basic_module[level](
                    torch.cat(
                        [
                            ref[level],
                            arch_util.flow_warp(
                                supp[level],
                                upsampled_flow.permute(0, 2, 3, 1),
                                interp_mode="bilinear",
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
                scale = 2 ** (
                    5 - level
                )  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = bilinear_upsample_2d(
                    flow,
                    size=(h // scale, w // scale),
                )
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)

                if torch.abs(flow_out).mean() > 200:
                    flow_out.clamp(-250, 250)

                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
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
