import torch
from torch import Tensor
from torchvision.ops import deform_conv2d  # type: ignore[import]

from bsrt.model.deform_sep_conv_net import DeformSepConvNet


class FlowGuidedDeformSepConvNet(DeformSepConvNet):
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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(  # type: ignore[override]
        self, input: Tensor, fea: Tensor, flows: Tensor
    ) -> Tensor:
        """input: input features for deformable conv: N, C, H, W.
        fea: other features used for generating offsets and mask: N, C, H, W.
        flows: N, 2, H, W.
        """
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        offset = torch.tanh(offset) * 10  # max_residue_magnitude
        offset = offset + flows.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        offset = offset.clamp(-100, 100)

        mask = torch.sigmoid(mask)
        conv: Tensor = deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )
        return conv
