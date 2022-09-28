from torch import Tensor
import math
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from metrics.utils.ignore_boundry import ignore_boundary

from option import Config


def reduce_mean(tensor: Tensor, nprocs: int) -> Tensor:
    rt: Tensor = tensor.clone()  # type: ignore
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # type: ignore
    rt /= nprocs
    return rt


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * torch.abs(dx) / 2.0
    loss_y = weights_y * torch.abs(dy) / 2.0
    return torch.mean(loss_x) / 2.0 + torch.mean(loss_y) / 2.0


def smooth_loss(flow, img):
    loss = smooth_grad_1st(flow, img, 10)
    return sum([torch.mean(loss)])


def cleanup():
    dist.destroy_process_group()


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = ignore_boundary(diff, shave)
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


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
        kwargs_optimizer["betas"] = config.betas
        kwargs_optimizer["eps"] = config.epsilon
    elif config.optimizer == "RMSprop":
        optimizer_class = optim.RMSprop
        kwargs_optimizer["eps"] = config.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), config.decay.split("-")))
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


def write_gray_to_tfboard(img):
    img_debug = img[0, ...].detach().cpu().numpy()

    # img_debug = cv2.normalize(img_debug, None, 0, 255,
    #                           cv2.NORM_MINMAX, cv2.CV_8U)
    img_debug = img_debug * 255
    img_debug = np.clip(img_debug, 0, 255)
    img_debug = img_debug.astype(np.uint8)
    return img_debug[0, ...]


######################## BayerUnifyAug ############################

BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
NORMALIZATION_MODE = ["crop", "pad"]


def bayer_unify(raw, input_pattern, target_pattern, mode) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif (
        input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]
    ):
        h_offset, w_offset = 1, 0
    elif (
        input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]
    ):
        h_offset, w_offset = 0, 1
    elif (
        input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]
    ):
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError("Unexpected pair of input and target bayer pattern!")

    if mode == "pad":
        # out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        out = F.pad(raw, (w_offset, w_offset, h_offset, h_offset), mode="reflect")
    elif mode == "crop":
        _, _, _, h, w = raw.shape
        out = raw[..., h_offset : h - h_offset, w_offset : w - w_offset]
    else:
        raise ValueError("Unknown normalization mode!")

    return out


def bayer_aug(
    raw, flip_h=False, flip_w=False, transpose=False, input_pattern="RGGB"
) -> np.ndarray:
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

    out = bayer_unify(out, aug_pattern, target_pattern, "crop")
    return out
