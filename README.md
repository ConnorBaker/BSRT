# BSRT: Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment (CVPRW 2022)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bsrt-improving-burst-super-resolution-with/burst-image-super-resolution-on-burstsr)](https://paperswithcode.com/sota/burst-image-super-resolution-on-burstsr?p=bsrt-improving-burst-super-resolution-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bsrt-improving-burst-super-resolution-with/burst-image-super-resolution-on)](https://paperswithcode.com/sota/burst-image-super-resolution-on?p=bsrt-improving-burst-super-resolution-with) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Algolzw/BSRT)

This work addresses the Burst Super-Resolution (BurstSR) task using a new architecture, which requires restoring a high-quality image from a sequence of noisy, misaligned, and low-resolution RAW bursts. To overcome the challenges in BurstSR, we propose a **B**urst **S**uper-**R**esolution **T**ransformer (**BSRT**), which can significantly improve the capability of extracting inter-frame information and reconstruction. To achieve this goal, we propose a Pyramid Flow-Guided Deformable Convolution Network (Pyramid FG-DCN) and incorporate Swin Transformer Blocks and Groups as our main backbone.  More specifically,  we combine optical flows and deformable convolutions, hence our BSRT can handle misalignment and aggregate the potential texture information in multi-frames more efficiently. In addition, our Transformer-based structure can capture long-range dependency to further improve the performance. The evaluation on both synthetic and real-world tracks demonstrates that our approach achieves a new state-of-the-art in BurstSR task. Further, our BSRT wins the championship in the NTIRE2022 Burst Super-Resolution Challenge.

Read the full paper on arXiv: <https://arxiv.org/abs/2204.08332>.

> **Note**
> BSRT is the winner of the NTIRE 2022 Burst Super-Resolution Challenge Real-World Track!
> You can also find our winner method in NTIRE 2021 Burst Super-Resolution Challenge [here](https://github.com/Algolzw/EBSR).

> **Warning**
> This fork of the repo is under heavy development. The code is likely to chance and no I provide no promises of stability. It follows that this README is also likely to change and should not be considered authoritative or even correct!

## Comparison with State-of-the-art Burst Super-Resolution Methods

![ts](figs/ts.png)

## Overview Architecture

![overview.png](figs/overview.png)

## Dependencies

- Docker: >= 20
- Python: Python 3.10
- CUDA: Runtime must be available

## Training

### Hyperparameter search

1. Create a ramdisk to store the training dataset

  ```bash
  mkdir "$HOME/ramdisk"
  sudo mount -t tmpfs tmpfs "$HOME/ramdisk"
  sudo chown $USER:$USER -R "$HOME/ramdisk"
  mkdir "$HOME/ramdisk/datasets"
  ```

2. Download and run the docker image

  ```bash
  docker run \
    --gpus all \
    --mount type=bind,source="$HOME/ramdisk/datasets",target=/datasets \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm connorbaker01/bsrt:main
  ```

3. Run a hyperparameter search backed by Optuna

  ```bash
  python -m bsrt.tuning.tuner \
    --experiment_name "model_with_adam_plateau" \
    --optimizer "AdamW" \
    --scheduler "ReduceLROnPlateau" \
    --precision "bfloat16" \
    --num_trials 1 \
    --max_epochs 20 \
    --batch_size 16 \
    --limit_train_batches 0.1 \
    --limit_val_batches 0.1 \
    --data_dir /datasets \
    --wandb_api_key="<your WandB API key>" \
    --db_uri="<your DB connection string>"
  ```

### Pre-training with synthetic data

TODO(connor)

Original parameters used for pre-training:

```bash
python main.py --n_GPUs 8 --print_every 40 --lr 0.0001 --decay 150-300 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 32 --burst_size 14 --patch_size 256
```

### Training with real-world data

TODO(connor)

Original parameters used for training:

```bash
python main.py --n_GPUs 8 --print_every 20 --lr 0.00005 --decay 40-80 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 8 --burst_size 14 --patch_size 80 --pre_train ../../synthetic/train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth 
```

## Inference

TODO(connor): add inference script and pretrained weights

## Results

### Comparison on Synthetic dataset

![cmp_syn.png](figs/cmp_syn.png)

### Comparison on Real-World dataset

![cmp_real.png](figs/cmp_real.png)

## Citations

If our code helps your research or work, please consider citing our paper.
The following is a BibTeX reference.

```text
@inproceedings{luo2022bsrt,
  title={BSRT: Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment},
  author={Luo, Ziwei and Li, Youwei and Cheng, Shen and Yu, Lei and Wu, Qi and Wen, Zhihong and Fan, Haoqiang and Sun, Jian and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={998--1008},
  year={2022}
}
```

## Contact

email: [ziwei.ro@gmail.com]
