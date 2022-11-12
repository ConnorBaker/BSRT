from typing import Sequence, Union

import torch
from torch import Tensor

from bsrt.model.non_local.non_local_general import _NonLocalBlockNDGeneral


class _NonLocalBlockND(_NonLocalBlockNDGeneral):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True,
    ):
        super().__init__(
            in_channels=in_channels,
            inter_channels=inter_channels,
            dimension=dimension,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
            kind="cross_dot_product",
        )

    def forward(self, x: Tensor, return_nl_map: bool = False) -> Union[Tensor, Sequence[Tensor]]:
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        assert self.inter_channels is not None
        batch_size = x.size(0)

        g_x: Tensor = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x: Tensor = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x: Tensor = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f: Tensor = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=1,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        sub_sample=True,
        bn_layer=True,
    ):
        super(NONLocalBlock3D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=3,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )
