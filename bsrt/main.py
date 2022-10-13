from model.bsrt import BSRT
from datasets.synthetic_train_zurich_raw2rgb_data_module import (
    SyntheticTrainZurichRaw2RgbDatasetDataModule,
)
import logging
import optuna
import os
import torch
import torch.backends.cuda
import torch.backends.cudnn
import logging
from pytorch_lightning.cli import LightningCLI

# configure logging at the root level of Lightning
logger = logging.getLogger("pytorch_lightning")
logger.setLevel(logging.INFO)


os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
os.environ["NCCL_SOCKET_NTHREADS"] = "4"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TORCH_CUDNN_V8_API_DEBUG"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def cli_main():
    cli = LightningCLI(BSRT, SyntheticTrainZurichRaw2RgbDatasetDataModule)


if __name__ == "__main__":
    cli_main()
