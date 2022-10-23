import torch
import torch.nn as nn

from .non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
from .non_local.non_local_dot_product import NONLocalBlock2D as NonLocal


class CrossNonLocalFusion(nn.Module):
    """Cross Non Local fusion module"""

    def __init__(self, nf=64, out_feat=96, nframes=5, center=2):
        super(CrossNonLocalFusion, self).__init__()
        self.center = center

        self.non_local_T = nn.ModuleList()
        self.non_local_F = nn.ModuleList()

        for i in range(nframes):
            self.non_local_T.append(
                NonLocalCross(
                    nf, inter_channels=nf // 2, sub_sample=True, bn_layer=False
                )
            )
            self.non_local_F.append(
                NonLocal(nf, inter_channels=nf // 2, sub_sample=True, bn_layer=False)
            )

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf * 2, out_feat, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        ref = aligned_fea[:, self.center, :, :, :].clone()

        cor_l = []
        non_l = []
        for i in range(N):
            nbr = aligned_fea[:, i, :, :, :]
            non_l.append(self.non_local_F[i](nbr))
            cor_l.append(self.non_local_T[i](nbr, ref))

        aligned_fea_T = torch.cat(cor_l, dim=1)
        aligned_fea_F = torch.cat(non_l, dim=1)
        aligned_fea = torch.cat([aligned_fea_T, aligned_fea_F], dim=1)

        #### fusion
        fea = self.fea_fusion(aligned_fea)

        return fea
