import torch
from torch import nn
from torchvision.ops import DeformConv2d, deform_conv2d


class DCN_sep(DeformConv2d):
    """Use other features to generate offsets and masks"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
    ):
        super(DCN_sep, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )
        channels_ = self.groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea):
        """input: input features for deformable conv
        fea: other features used for generating offsets and mask"""
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        offset = torch.clamp(offset, -100, 100)

        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


class FlowGuidedDCN(DeformConv2d):
    """Use other features to generate offsets and masks"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
    ):
        super(FlowGuidedDCN, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )
        channels_ = self.groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels, channels_, kernel_size, stride, padding, bias=True
        )

        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea, flows):
        """input: input features for deformable conv: N, C, H, W.
        fea: other features used for generating offsets and mask: N, C, H, W.
        flows: N, 2, H, W.
        """
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = torch.tanh(torch.cat((o1, o2), dim=1)) * 10  # max_residue_magnitude
        offset = offset + flows.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # offset_mean = torch.mean(torch.abs(offset))
        # if offset_mean > 250:
        # TODO: Is this bad? Should we clamp or terminate the experiment?
        # print(
        #     "FlowGuidedDCN: Offset mean is {}, larger than 100.".format(offset_mean)
        # )
        # offset = offset.clamp(-50, 50)
        # return None

        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )
