from typing import cast
from utils.types import BayerPattern, NormalizationMode
from metrics.aligned.l1 import AlignedL1
from metrics.aligned.psnr import AlignedPSNR
from metrics.charbonnier_loss import CharbonnierLoss
from metrics.l1 import L1
from metrics.l2 import L2
from metrics.ms_ssim_loss import MSSSIMLoss
from metrics.psnr import PSNR
from model.bsrt import BSRT
from option import Config, LossName, DataTypeName
from torch import Tensor
from torchmetrics.metric import Metric
from typing_extensions import Literal, get_args, overload
from utils.postprocessing_functions import BurstSRPostProcess, SimplePostProcess
import os
import torch
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


def make_model(config: Config) -> BSRT:
    nframes = config.burst_size
    img_size = config.patch_size * 2
    # FIXME: This overrides below?
    patch_size = 1
    print("FIXME: Patch size is being ignored!")
    in_chans = config.burst_channel
    out_chans = config.n_colors

    if config.model_level == "S":
        depths = [6] * 1 + [6] * 4
        num_heads = [6] * 1 + [6] * 4
        embed_dim = 60
    elif config.model_level == "L":
        depths = [6] * 1 + [8] * 6
        num_heads = [6] * 1 + [6] * 6
        embed_dim = 180
    window_size = 8
    mlp_ratio = 2
    upscale = config.scale
    non_local = config.non_local
    use_swin_checkpoint = config.use_checkpoint

    bsrt = BSRT(
        config=config,
        nframes=nframes,
        img_size=img_size,
        # FIXME: Is this overriden above?
        patch_size=patch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,  # type: ignore
        depths=depths,  # type: ignore
        num_heads=num_heads,  # type: ignore
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        upscale=upscale,
        non_local=non_local,
        use_swin_checkpoint=use_swin_checkpoint,
    )

    return bsrt


def make_optimizer(config: Config, target):
    """
    make optimizer and scheduler together
    """
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {"lr": config.lr, "weight_decay": config.weight_decay}

    if config.optimizer == "SGD":
        optimizer_class = optim.SGD
        kwargs_optimizer["momentum"] = config.momentum
    elif config.optimizer == "ADAM":
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (config.beta_gradient, config.beta_square)
        kwargs_optimizer["eps"] = config.epsilon
    elif config.optimizer == "RMSprop":
        optimizer_class = optim.RMSprop
        kwargs_optimizer["eps"] = config.epsilon

    # scheduler
    milestones = list(map(int, config.decay_milestones))
    kwargs_scheduler = {"milestones": milestones, "gamma": config.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch):
                    self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, "optimizer.pt")

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


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
