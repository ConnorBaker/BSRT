import torch
import torch.nn as nn
from torch import Tensor

from bsrt.model.deform_sep_conv_net import DeformSepConvNet
from bsrt.model.flow_guided_deform_sep_conv_net import FlowGuidedDeformSepConvNet
from bsrt.utils.bilinear_upsample_2d import bilinear_upsample_2d


class FlowGuidedPCDAlign(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    """

    def __init__(self, nf: int = 64, groups: int = 8) -> None:
        super(FlowGuidedPCDAlign, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = FlowGuidedDeformSepConvNet(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)

        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = FlowGuidedDeformSepConvNet(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = FlowGuidedDeformSepConvNet(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DeformSepConvNet(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(  # type: ignore[override]
        self, nbr_fea_l: Tensor, nbr_fea_warped_l: Tensor, ref_fea_l: Tensor, flows_l: Tensor
    ) -> Tensor:
        """align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        """
        # L3
        L3_offset: Tensor = torch.cat([nbr_fea_warped_l[2], ref_fea_l[2], flows_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea: Tensor = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset, flows_l[2]))
        # L2
        L3_offset = bilinear_upsample_2d(L3_offset, scale_factor=2)
        L2_offset: Tensor = torch.cat([nbr_fea_warped_l[1], ref_fea_l[1], flows_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea: Tensor = self.L2_dcnpack(nbr_fea_l[1], L2_offset, flows_l[1])
        L3_fea = bilinear_upsample_2d(
            L3_fea,
            scale_factor=2,
        )
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L2_offset = bilinear_upsample_2d(
            L2_offset,
            scale_factor=2,
        )
        L1_offset: Tensor = torch.cat([nbr_fea_warped_l[0], ref_fea_l[0], flows_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea: Tensor = self.L1_dcnpack(nbr_fea_l[0], L1_offset, flows_l[0])
        L2_fea = bilinear_upsample_2d(L2_fea, scale_factor=2)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        offset: Tensor = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.cas_dcnpack(L1_fea, offset)

        return L1_fea
