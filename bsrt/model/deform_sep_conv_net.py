import torch
from torch import Tensor, nn
from torchvision.ops import DeformConv2d, deform_conv2d


class DeformSepConvNet(DeformConv2d):
    """Use other features to generate offsets and masks"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        channels_ = self.groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels_,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        assert self.conv_offset_mask.bias is not None
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input: Tensor, fea: Tensor) -> Tensor:
        """input: input features for deformable conv: N, C, H, W.
        fea: other features used for generating offsets and mask: N, C, H, W.
        """
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        offset = offset.clamp(-100, 100)

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
