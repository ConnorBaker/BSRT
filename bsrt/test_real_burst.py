from datasets.burstsr_dataset import flatten_raw_image_batch, pack_raw_image_batch
from datasets.synthetic_burst_test_set import SyntheticBurstTest
from datasets.realworld_burst_test_set import RealWorldBurstTest
from option import config, Config
from tqdm import tqdm
import cv2
import model
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data.distributed
import utility


checkpoint = utility.checkpoint(config)


def ttaup(burst):
    burst0 = flatten_raw_image_batch(burst)  # B, T, C, H, W
    burst1 = utility.bayer_aug(burst0, flip_h=False, flip_w=False, transpose=True)
    burst0 = pack_raw_image_batch(burst0)
    burst1 = pack_raw_image_batch(burst1)

    return [burst0, burst1]


def ttadown(bursts):
    burst0 = bursts[0]
    burst1 = bursts[1].permute(0, 1, 3, 2)
    out = (burst0 + burst1) / 2
    return out


def main():
    mp.spawn(main_worker, nprocs=1, args=(1, config))


def main_worker(local_rank: int, nprocs: int, config: Config):
    device = "cuda"
    cudnn.benchmark = True
    config.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    if config.data_type == "synthetic":
        dataset = SyntheticBurstTest(config.root)
        out_dir = "bsrt_synburst"
    else:
        dataset = RealWorldBurstTest(config.root)
        out_dir = "bsrt_realworld"

    os.makedirs(out_dir, exist_ok=True)

    _model = model.Model(config, checkpoint)

    tt = []
    for idx in tqdm(range(len(dataset))):
        burst, meta_info = dataset[idx]
        burst_name = meta_info["burst_name"]

        burst = burst.unsqueeze(0)
        if config.data_type == "synthetic":
            bursts = ttaup(burst)
            srs = []
            with torch.no_grad():
                for x in bursts:
                    tic = time.time()
                    sr = _model(x, 0)
                    toc = time.time()
                    tt.append(toc - tic)
                    srs.append(sr)

            sr = ttadown(srs)
        else:
            with torch.no_grad():
                tic = time.time()
                sr = _model(burst, 0).float()
                toc = time.time()
                tt.append(toc - tic)

        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (
            (sr.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2**14)
            .cpu()
            .numpy()
            .astype(np.uint16)
        )
        cv2.imwrite("{}/{}.png".format(out_dir, burst_name), net_pred_np)

    print("avg time: {:.4f}".format(np.mean(tt)))
    utility.cleanup()


if __name__ == "__main__":
    main()
