# Overall

- Instructions for training and testing
- TODO:
    - [ ] We should not use the validation dataset for testing!
    - [ ] Investigate impact of using `--use_checkpoint`

## Setup

- Grab a copy of the zurich-raw-to-rgb (ZRR) dataset: <https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip>
- Extract it to `~/datasets/zurich-raw-to-rgb`
    - The path `~/datasets/zurich-raw-to-rgb/train/canon` should exist contain thousands of `.jpg` files
- Split it into train and test
    - [ ] Describe the size of the split
- Grab a copy of the synthetic burst validation dataset: <https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip>
- Extract it to `~/datasets/SyntheticBurstVal/val`
    - The paths `~/datasets/SyntheticBurstVal/val/bursts` and `~/datasets/SyntheticBurstVal/val/gt` should exist and contain the folders `0000` through `0299`, each of which contains 14 `.png` files (in the case of `bursts`) or one `.png` file and one metadata `.pkl` file (in the case of `gt`)

## Training synthetic

- Main has much higher learning rate (1e-4 vs 3e-4)
- Higher decay (100-200 vs 50-100)

### Main readme

```bash
python main.py --n_GPUs 1 --print_every 40 --lr 0.0001 --decay 100-200 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 48 --burst_size 14 --patch_size 256 --root ~/datasets/zurich-raw-to-rgb  --val_root ~/datasets/SyntheticBurstVal --models_root ~/models --epochs 100 --use_checkpoint --save_models
```

### Synthetic readme

```bash
python main.py --n_GPUs 1 --print_every 40 --lr 0.00003 --decay 50-100 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 48 --burst_size 14 --patch_size 256 --root ~/datasets/zurich-raw-to-rgb  --val_root ~/datasets/SyntheticBurstVal --models_root ~/models --epochs 100 --use_checkpoint --save_models
```

## Testing synthetic

- Main does not specify `--fp16`

### Main readme

```bash
python test_synburst.py --n_GPUs 1 --model BSRT --model_level S --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth --root ~/datasets/SyntheticBurstVal --save_results --save_gt
```

### Synthetic readme

```bash
python test_synburst.py --n_GPUs 1 --model BSRT --model_level S --fp16 --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth --root ~/datasets/SyntheticBurstVal --save_results --save_gt
```
