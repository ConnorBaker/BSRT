from argparse import Namespace
from datasets.burstsr_dataset import BurstSRDataset
from datasets.synthetic_burst_val_set import SyntheticBurstVal
from option import args
from pwcnet.pwcnet import PWCNet
from tqdm import tqdm
from utils.metrics import AlignedPSNR, PSNR
from utils.postprocessing_functions import BurstSRPostProcess, SimplePostProcess
import cv2
import model
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data.distributed
import utility


checkpoint = utility.checkpoint(args)


def main() -> None:
    mp.spawn(main_worker, nprocs=1, args=(1, args)) # type: ignore


def main_worker(local_rank: int, nprocs: int, args: Namespace) -> None:
    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank) # type: ignore

    _model: : nn.Module = model.Model(args, checkpoint)

    for param in _model.parameters():
        param.requires_grad = False

    if args.data_type == "synthetic":
        dataset = SyntheticBurstVal(root=args.root)
        out_dir = "val/bsrt_synburst"
        psnr_fn = PSNR(boundary_ignore=40)
        postprocess_fn = SimplePostProcess(return_np=True)

    elif args.data_type == "real":
        dataset = BurstSRDataset(root=args.root, burst_size=14, crop_sz=80, split="val")
        out_dir = "val/bsrt_real"
        alignment_net: nn.Module = PWCNet(
            load_pretrained=True,
            weights_path=args.models_root + "/pwcnet-network-default.pth",
        ).cuda()  # type: ignore
        for param in alignment_net.parameters():
            param.requires_grad = False

        psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)
        postprocess_fn = BurstSRPostProcess(return_np=True)

    os.makedirs(out_dir, exist_ok=True)

    tt: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    for idx in tqdm(range(len(dataset))):
        burst_, gt, meta_info = dataset[idx][:3]
        burst_ = burst_.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()
        name = meta_info["burst_name"]

        with torch.no_grad():
            tic = time.time()
            sr = _model(burst_, 0).float()
            toc = time.time()
            tt.append(toc - tic)

        if args.data_type == "synthetic":
            psnr, ssim, lpips = psnr_fn(sr, gt)

        elif args.data_type == "real":
            psnr, ssim, lpips = psnr_fn(sr, gt, burst_)

        psnrs.append(psnr.item())
        ssims.append(ssim.item())
        lpipss.append(lpips.item())

        lrs = burst_[0]
        os.makedirs(f"{out_dir}/{name}", exist_ok=True)
        for i, lr in enumerate(lrs):
            lr = postprocess_fn.process(lr[[0, 1, 3], ...], meta_info)
            lr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
            cv2.imwrite("{}/{}/{:2d}.png".format(out_dir, name, i), lr)

        gt = postprocess_fn.process(gt[0], meta_info)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        cv2.imwrite("{}/{}_gt.png".format(out_dir, name), gt)

        sr_ = postprocess_fn.process(sr[0], meta_info)
        sr_ = cv2.cvtColor(sr_, cv2.COLOR_RGB2BGR)
        cv2.imwrite("{}/{}_bsrt.png".format(out_dir, name), sr_)

        del burst_
        del sr
        del gt

    print(f"avg PSNR: {np.mean(psnrs):.6f}") # type: ignore
    print(f"avg SSIM: {np.mean(ssims):.6f}") # type: ignore
    print(f"avg LPIPS: {np.mean(lpipss):.6f}") # type: ignore
    print(f"avg time: {np.mean(tt):.6f}") # type: ignore

    # utility.cleanup()


if __name__ == "__main__":
    main()
