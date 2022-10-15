from datasets.synthetic_train_zurich_raw2rgb_data_module import (
    SyntheticTrainZurichRaw2RgbDatasetDataModule,
)
from model.bsrt import BSRT
from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    import os

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["TORCH_CUDNN_V8_API_DEBUG"] = "1"
    import torch
    import torch.backends.cuda
    import torch.backends.cudnn

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    LightningCLI(
        BSRT, SyntheticTrainZurichRaw2RgbDatasetDataModule, save_config_callback=None
    )
