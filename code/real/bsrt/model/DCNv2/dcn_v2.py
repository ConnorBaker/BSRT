# Taken with much thanks from the Torchvision project, and modified slightly.
# The original was taken from:
# https://github.com/pytorch/vision/blob/93c85bbcc31f8d5a052daf06f2f91f39697af1a4/torchvision/ops/deform_conv.py
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    n_weight_grps: int,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    # This is now unecessary because we don't need to calculate n_weight_grps.
    # See comment below for more.
    # _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    # Since we don't quotient the second dimension of the weight by the number
    # of groups, there's no way to get it back out by doing this quotient.
    # Instead, we take as a function argument.
    # n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )


class DeformConv2d(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        # Since we're loading from a pre-trained model to avoid wasting days/
        # weeks, we need to keep the second argument to torch.empty as-is.
        # The deform_conv2d function used the fact that we had divided the
        # second dimension to retrieve the number of groups. We instead pass
        # the number of groups as an argument directly to the function.
        # self.weight = Parameter(
        #     torch.empty(
        #         out_channels,
        #         in_channels // groups,
        #         self.kernel_size[0],
        #         self.kernel_size[1],
        #     )
        # )
        self.weight = Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
                masks to be applied for each position in the convolution kernel.
        """
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}" if self.padding != (0, 0) else ""
        s += f", dilation={self.dilation}" if self.dilation != (1, 1) else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"

        return s


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
        # offset = torch.clamp(offset, -100, 100)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 250:
            print("Offset mean is {}, larger than 100.".format(offset_mean))
            # return None
            # offset[offset>=150] = 1e-3
            # offset = offset.clamp(-50, 50)

        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.groups,
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

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 250:
            print(
                "FlowGuidedDCN: Offset mean is {}, larger than 100.".format(offset_mean)
            )
            # offset = offset.clamp(-50, 50)
            # return None

        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.groups,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )
