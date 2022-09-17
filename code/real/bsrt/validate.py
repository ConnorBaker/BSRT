from argparse import Namespace
from datasets.burstsr_dataset import BurstSRDataset
from option import args
from pwcnet.pwcnet import PWCNet
from tqdm import tqdm
from utils.metrics import AlignedPSNR
import model
import numpy as np
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

    dataset = BurstSRDataset(root=args.root, burst_size=14, crop_sz=80, split="val")

    _model: nn.Module = model.Model(args, checkpoint)
    for param in _model.parameters():
        param.requires_grad = False

    alignment_net: nn.Module = PWCNet(
        load_pretrained=True,
        weights_path="/home/ubuntu/bsrt/models/pwcnet-network-default.pth",
    ).cuda()  # type: ignore
    for param in alignment_net.parameters():
        param.requires_grad = False

    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)

    tt: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    for idx in tqdm(range(len(dataset))):
        burst, gt, _meta_info_burst, _meta_info_gt = dataset[idx]
        burst = burst.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()

        with torch.no_grad():
            tic = time.time()
            sr = _model(burst, 0).float()
            toc = time.time()
            tt.append(toc - tic)

            psnr, ssim, lpips = aligned_psnr_fn(sr, gt, burst)
            psnrs.append(psnr.item())
            ssims.append(ssim.item())
            lpipss.append(lpips.item())

        del burst
        del sr
        del gt

    print(f"avg PSNR: {np.mean(psnrs):.6f}") # type: ignore
    print(f"avg SSIM: {np.mean(ssims):.6f}") # type: ignore
    print(f"avg LPIPS: {np.mean(lpipss):.6f}") # type: ignore
    print(f"avg time: {np.mean(tt):.6f}") # type: ignore

    # utility.cleanup()


if __name__ == "__main__":
    main()
