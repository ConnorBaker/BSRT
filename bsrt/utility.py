from typing import cast
from utils.types import BayerPattern, NormalizationMode
from metrics.aligned.l1 import AlignedL1
from metrics.aligned.psnr import AlignedPSNR
from metrics.charbonnier_loss import CharbonnierLoss
from metrics.l1 import L1
from metrics.l2 import L2
from metrics.ms_ssim_loss import MSSSIMLoss
from metrics.psnr import PSNR
from option import LossName, DataTypeName
from torch import Tensor
from torchmetrics.metric import Metric
from typing_extensions import Literal, get_args, overload
from utils.postprocessing_functions import BurstSRPostProcess, SimplePostProcess
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def make_loss_fn(loss: LossName, data_type: DataTypeName) -> Metric:
    match loss:
        case "L1":
            match data_type:
                case "synthetic":
                    return L1()
                case "real":
                    # FIXME: Reduce duplication with make_psnr_fn by using the same alignment_net
                    from pwcnet.pwcnet import PWCNet

                    alignment_net = PWCNet()
                    for param in alignment_net.parameters():
                        param.requires_grad = False
                    return AlignedL1(alignment_net=alignment_net, boundary_ignore=40)
        case "MSE":
            return L2()
        case "CB":
            return CharbonnierLoss()
        case "MSSSIM":
            return MSSSIMLoss()


@overload
def make_postprocess_fn(data_type: Literal["synthetic"]) -> SimplePostProcess:
    ...


@overload
def make_postprocess_fn(data_type: Literal["real"]) -> BurstSRPostProcess:
    ...


def make_postprocess_fn(
    data_type: DataTypeName,
) -> BurstSRPostProcess | SimplePostProcess:
    match data_type:
        case "synthetic":
            return SimplePostProcess(return_np=True)
        case "real":
            return BurstSRPostProcess(return_np=True)


def make_psnr_fn(data_type: DataTypeName) -> Metric:
    match data_type:
        case "synthetic":
            return PSNR(boundary_ignore=40)
        case "real":
            from pwcnet.pwcnet import PWCNet

            alignment_net = PWCNet()
            for param in alignment_net.parameters():
                param.requires_grad = False
            return AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)

######################## BayerUnifyAug ############################


def bayer_unify(
    raw: Tensor,
    input_pattern: BayerPattern,
    target_pattern: BayerPattern,
    mode: NormalizationMode,
) -> Tensor:
    """
    Convert a bayer raw image from one bayer pattern to another.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """

    if input_pattern == target_pattern:
        # A match!
        h_offset, w_offset = 0, 0
    elif input_pattern[0:2] == target_pattern[2:4]:
        # Channels are rotated
        h_offset, w_offset = 1, 0
    elif (
        input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]
    ):
        # Channels are flipped
        h_offset, w_offset = 0, 1
    elif input_pattern[0:2] == target_pattern[3:1:-1]:
        # Channels are rotated and flipped
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError("Unexpected pair of input and target bayer pattern!")

    if mode == "pad":
        out = F.pad(raw, (w_offset, w_offset, h_offset, h_offset), mode="reflect")
    elif mode == "crop":
        _, _, _, h, w = raw.shape
        out = raw[..., h_offset : h - h_offset, w_offset : w - w_offset]

    return out


def bayer_aug(
    raw: Tensor,
    flip_h: bool = False,
    flip_w: bool = False,
    transpose: bool = False,
    input_pattern: BayerPattern = "RGGB",
) -> Tensor:
    """
    Apply augmentation to a bayer raw image.
    """

    aug_pattern, target_pattern = input_pattern, input_pattern

    out = raw
    if flip_h:
        out = torch.flip(out, [3])  # GBRG, RGGB
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out = torch.flip(out, [4])
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out = out.permute(0, 1, 2, 4, 3)
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

    assert aug_pattern in get_args(BayerPattern)
    aug_pattern = cast(BayerPattern, aug_pattern)
    out = bayer_unify(out, aug_pattern, target_pattern, "crop")
    return out
