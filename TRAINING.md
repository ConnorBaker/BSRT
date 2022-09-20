# Overall

- Instructions for training and testing
- TODO:
    - [ ] We should not use the validation dataset for testing!
    - [ ] Investigate impact of using `--use_checkpoint`

## Setup

- Grab a copy of the zurich-raw-to-rgb (ZRR) dataset: <https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip>
- Extract it to `~/working/datasets/zurich-raw-to-rgb`
    - The path `~/working/datasets/zurich-raw-to-rgb/train/canon` should exist contain thousands of `.jpg` files
- Split it into train and test
    - [ ] Describe the size of the split
- Grab a copy of the synthetic burst validation dataset: <https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip>
- Extract it to `~/working/datasets/SyntheticBurstVal/val`
    - The paths `~/working/datasets/SyntheticBurstVal/val/bursts` and `~/working/datasets/SyntheticBurstVal/val/gt` should exist and contain the folders `0000` through `0299`, each of which contains 14 `.png` files (in the case of `bursts`) or one `.png` file and one metadata `.pkl` file (in the case of `gt`)

## Training synthetic

- Main has much higher learning rate (1e-4 vs 3e-4)
- Higher decay (100-200 vs 50-100)

### Main readme

```bash
python main.py --n_GPUs 1 --print_every 40 --lr 0.0001 --decay 100-200 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 48 --burst_size 14 --patch_size 256 --root ~/working/datasets/zurich-raw-to-rgb --val_root ~/working/datasets/SyntheticBurstVal --models_root ~/working/models --epochs 100 --use_checkpoint --save_models
```

### Synthetic readme

```bash
python main.py --n_GPUs 1 --print_every 40 --lr 0.00003 --decay 50-100 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 48 --burst_size 14 --patch_size 256 --root ~/working/datasets/zurich-raw-to-rgb --val_root ~/working/datasets/SyntheticBurstVal --models_root ~/working/models --epochs 100 --use_checkpoint --save_models
```

## Testing synthetic

- Main does not specify `--fp16`

### Main readme

```bash
python test_synburst.py --n_GPUs 1 --model BSRT --model_level S --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth --root ~/working/datasets/SyntheticBurstVal --models_root ~/working/models --save_results --save_gt
```

### Synthetic readme

```bash
python test_synburst.py --n_GPUs 1 --model BSRT --model_level S --fp16 --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth --root ~/working/datasets/SyntheticBurstVal --models_root ~/working/models --save_results --save_gt
```

## Observations on batch size vs. epoch time

Results of training for one epoch on LambdaLabs' `gpu.8x.a100` instance.

From these results it would appear that the fastest way to reach 300 epochs on this machine is to use a batch size of 64, with 48 being a close second.

However, this is to say nothing of which batch size provides the best PSNR.

### Batch size of 32

```text
[1280/46839]	[0.0505]	23.7+57.7s
[2560/46839]	[0.0397]	15.6+0.0s
[3840/46839]	[0.0373]	15.7+0.0s
[5120/46839]	[0.0316]	15.8+0.0s
[6400/46839]	[0.0293]	15.6+0.0s
save model...
[7680/46839]	[0.0216]	15.6+0.1s
[8960/46839]	[0.0263]	15.7+0.0s
[10240/46839]	[0.0220]	15.5+0.0s
[11520/46839]	[0.0198]	15.6+0.0s
[12800/46839]	[0.0291]	15.4+0.0s
save model...
[14080/46839]	[0.0290]	15.3+0.1s
[15360/46839]	[0.0223]	15.4+0.0s
[16640/46839]	[0.0174]	15.6+0.0s
[17920/46839]	[0.0192]	15.6+0.0s
[19200/46839]	[0.0172]	15.6+0.0s
save model...
[20480/46839]	[0.0188]	15.5+0.1s
[21760/46839]	[0.0220]	15.6+0.0s
[23040/46839]	[0.0172]	15.3+0.0s
[24320/46839]	[0.0210]	15.5+0.0s
[25600/46839]	[0.0248]	15.5+0.0s
save model...
[26880/46839]	[0.0172]	15.5+0.1s
[28160/46839]	[0.0184]	15.4+0.0s
[29440/46839]	[0.0183]	15.3+0.0s
[30720/46839]	[0.0154]	15.5+0.0s
[32000/46839]	[0.0174]	15.5+0.0s
save model...
[33280/46839]	[0.0208]	16.2+0.1s
[34560/46839]	[0.0212]	16.1+0.0s
[35840/46839]	[0.0147]	16.0+0.0s
[37120/46839]	[0.0166]	16.1+0.0s
[38400/46839]	[0.0223]	16.0+0.0s
save model...
[39680/46839]	[0.0195]	16.1+0.1s
[40960/46839]	[0.0203]	16.0+0.0s
[42240/46839]	[0.0180]	16.1+0.0s
[43520/46839]	[0.0154]	16.1+0.0s
[44800/46839]	[0.0158]	16.2+0.0s
save model...
[46080/46839]	[0.0194]	14.9+0.1s
0it [00:00, ?it/s]Epoch 1 cost time: 644.5s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:12,  3.09it/s]
38it [00:12,  3.08it/s]
38it [00:12,  3.13it/s]
38it [00:12,  3.07it/s]
38it [00:12,  3.14it/s]
[Epoch: 1]
[PSNR: 32.6025][SSIM: 0.8442][LPIPS: 0.2765][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:12,  3.04it/s]
38it [00:12,  3.12it/s]
38it [00:12,  3.09it/s]
Forward: 12.27s
```

Estimated runtime for 300 epochs: 644.5 sec * 300 epochs / (3600 sec / hour) = 53.71 hours

### Batch size of 48

```text
[1920/46839]	[0.0419]	35.5+100.3s
[3840/46839]	[0.0369]	18.6+0.0s
[5760/46839]	[0.0352]	18.7+0.0s
[7680/46839]	[0.0268]	18.9+0.0s
[9600/46839]	[0.0282]	18.8+0.0s
save model...
[11520/46839]	[0.0232]	18.7+0.1s
[13440/46839]	[0.0188]	18.6+0.0s
[15360/46839]	[0.0239]	18.5+0.0s
[17280/46839]	[0.0229]	18.4+0.0s
[19200/46839]	[0.0236]	18.6+0.0s
save model...
[21120/46839]	[0.0216]	18.5+0.1s
[23040/46839]	[0.0176]	18.4+0.0s
[24960/46839]	[0.0188]	18.4+0.0s
[26880/46839]	[0.0190]	18.6+0.0s
[28800/46839]	[0.0192]	18.4+0.0s
save model...
[30720/46839]	[0.0171]	18.5+0.1s
[32640/46839]	[0.0176]	18.4+0.0s
[34560/46839]	[0.0231]	18.4+0.0s
[36480/46839]	[0.0206]	18.4+0.0s
[38400/46839]	[0.0218]	18.6+0.0s
save model...
[40320/46839]	[0.0174]	18.4+0.1s
[42240/46839]	[0.0161]	18.5+0.0s
[44160/46839]	[0.0162]	17.7+0.0s
[46080/46839]	[0.0193]	17.7+0.0s
Epoch 1 cost time: 574.3s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:17,  2.20it/s]
38it [00:17,  2.21it/s]
38it [00:17,  2.23it/s]
38it [00:16,  2.24it/s]
38it [00:17,  2.23it/s]
38it [00:17,  2.21it/s]
[Epoch: 1]
[PSNR: 32.4249][SSIM: 0.8377][LPIPS: 0.2796][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:17,  2.23it/s]
38it [00:17,  2.22it/s]
Forward: 17.34s
```

Estimated runtime for 300 epochs: 574.3 sec * 300 epochs / (3600 sec / hour) = 47.86 hours

### Batch size of 64

```text
[2560/46839]	[0.0507]	32.5+115.0s
[5120/46839]	[0.0413]	20.1+0.0s
[7680/46839]	[0.0303]	20.2+0.0s
[10240/46839]	[0.0251]	20.0+0.1s
[12800/46839]	[0.0284]	20.1+0.0s
save model...
[15360/46839]	[0.0256]	19.9+0.1s
[17920/46839]	[0.0259]	20.0+0.0s
[20480/46839]	[0.0215]	20.1+0.0s
[23040/46839]	[0.0210]	20.2+0.0s
[25600/46839]	[0.0267]	20.2+0.0s
save model...
[28160/46839]	[0.0197]	20.1+0.1s
[30720/46839]	[0.0204]	20.1+0.0s
[33280/46839]	[0.0207]	20.0+0.0s
[35840/46839]	[0.0216]	20.3+0.0s
[38400/46839]	[0.0201]	20.2+0.1s
save model...
[40960/46839]	[0.0187]	19.7+0.1s
[43520/46839]	[0.0221]	19.7+0.0s
[46080/46839]	[0.0203]	19.7+0.0s
Epoch 1 cost time: 501.9s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:08,  4.55it/s]
38it [00:08,  4.57it/s]
38it [00:08,  4.38it/s]
38it [00:08,  4.52it/s]
38it [00:08,  4.48it/s]
38it [00:08,  4.50it/s]
[Epoch: 1]
[PSNR: 31.8445][SSIM: 0.8249][LPIPS: 0.2821][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:08,  4.41it/s]
38it [00:08,  4.51it/s]
Forward: 8.57s
```

Estimated runtime for 300 epochs: 501.9 sec * 300 epochs / (3600 sec / hour) = 41.83 hours

#### With 8xT4

The time to run one epoch on 8xT4:

```text
Fix keys: ['spynet', 'dcnpack'] for the first 5 epochs.
[2560/46839]	[0.0507]	75.4+115.4s
[5120/46839]	[0.0413]	59.6+0.1s
[7680/46839]	[0.0302]	59.7+0.1s
[10240/46839]	[0.0251]	57.6+0.1s
[12800/46839]	[0.0283]	57.1+0.1s
save model...
[15360/46839]	[0.0256]	57.3+0.2s
[17920/46839]	[0.0259]	57.9+0.1s
[20480/46839]	[0.0214]	59.2+0.1s
[23040/46839]	[0.0203]	60.3+0.1s
[25600/46839]	[0.0268]	59.4+0.1s
save model...
[28160/46839]	[0.0203]	59.5+0.2s
[30720/46839]	[0.0205]	59.4+0.1s
[33280/46839]	[0.0216]	57.4+0.1s
[35840/46839]	[0.0210]	57.3+0.1s
[38400/46839]	[0.0200]	57.4+0.1s
save model...
[40960/46839]	[0.0187]	57.7+0.2s
[43520/46839]	[0.0216]	59.0+0.1s
[46080/46839]	[0.0200]	59.5+0.1s
Epoch 1 cost time: 1209.7s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:10,  3.58it/s]
38it [00:10,  3.59it/s]
38it [00:10,  3.58it/s]
38it [00:10,  3.59it/s]
[Epoch: 1]
[PSNR: 32.0884][SSIM: 0.8278][LPIPS: 0.2822][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:10,  3.55it/s]
38it [00:10,  3.56it/s]
38it [00:10,  3.56it/s]
38it [00:10,  3.56it/s]
Forward: 10.70s
```

Estimated runtime for 300 epochs: 1209.7 sec * 300 epochs / (3600 sec / hour) = 100.81 hours

### Batch size of 80

```text
[3200/46839]	[0.0507]	60.8+159.1s
[6400/46839]	[0.0316]	23.0+0.0s
[9600/46839]	[0.0329]	23.1+0.0s
[12800/46839]	[0.0312]	23.0+0.0s
[16000/46839]	[0.0278]	23.0+0.0s
save model...
[19200/46839]	[0.0295]	23.0+0.1s
[22400/46839]	[0.0225]	23.0+0.0s
[25600/46839]	[0.0267]	23.0+0.0s
[28800/46839]	[0.0245]	23.0+0.0s
[32000/46839]	[0.0198]	23.0+0.0s
save model...
[35200/46839]	[0.0207]	22.8+0.1s
[38400/46839]	[0.0204]	22.6+0.0s
[41600/46839]	[0.0189]	22.6+0.0s
[44800/46839]	[0.0182]	22.7+0.0s
0it [00:00, ?it/s]Epoch 1 cost time: 543.2s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:15,  2.51it/s]
38it [00:15,  2.48it/s]
38it [00:15,  2.51it/s]
38it [00:15,  2.53it/s]
38it [00:15,  2.52it/s]
38it [00:15,  2.52it/s]
[Epoch: 1]
[PSNR: 31.8643][SSIM: 0.8225][LPIPS: 0.2864][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:15,  2.46it/s]
38it [00:15,  2.52it/s]
Forward: 15.17s
```

Estimated runtime for 300 epochs: 543.2 sec * 300 epochs / (3600 sec / hour) = 45.27 hours

### Batch size of 96

```text
[3840/46839]	[0.0504]	55.1+215.6s
[7680/46839]	[0.0347]	26.6+0.0s
[11520/46839]	[0.0308]	26.4+0.1s
[15360/46839]	[0.0288]	26.5+0.1s
[19200/46839]	[0.0269]	26.5+0.1s
save model...
[23040/46839]	[0.0254]	26.5+0.1s
[26880/46839]	[0.0234]	26.4+0.1s
[30720/46839]	[0.0234]	25.9+0.1s
[34560/46839]	[0.0238]	25.5+0.0s
[38400/46839]	[0.0197]	25.6+0.0s
save model...
[42240/46839]	[0.0182]	25.6+0.1s
[46080/46839]	[0.0215]	25.6+0.0s
Epoch 1 cost time: 575.0s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:14,  2.57it/s]
38it [00:14,  2.61it/s]
38it [00:14,  2.64it/s]
38it [00:14,  2.64it/s]
38it [00:14,  2.59it/s]
[Epoch: 1]
[PSNR: 31.7533][SSIM: 0.8160][LPIPS: 0.2910][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:14,  2.62it/s]
38it [00:14,  2.62it/s]
38it [00:14,  2.62it/s]
Forward: 14.78s
```

Estimated runtime for 300 epochs: 575.0 sec * 300 epochs / (3600 sec / hour) = 47.92 hours

### Batch size of 128

```text
[5120/46839]	[0.0551]	62.7+240.3s
[10240/46839]	[0.0296]	32.7+0.1s
[15360/46839]	[0.0334]	32.4+0.1s
[20480/46839]	[0.0279]	31.8+0.1s
[25600/46839]	[0.0287]	31.8+0.1s
save model...
[30720/46839]	[0.0248]	31.9+0.1s
[35840/46839]	[0.0237]	31.9+0.1s
[40960/46839]	[0.0236]	31.9+0.1s
[46080/46839]	[0.0222]	31.7+0.1s
Epoch 1 cost time: 578.4s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:07,  4.87it/s]
38it [00:08,  4.59it/s]
38it [00:07,  4.86it/s]
38it [00:08,  4.74it/s]
38it [00:07,  4.82it/s]
38it [00:08,  4.66it/s]
[Epoch: 1]
[PSNR: 31.1096][SSIM: 0.7941][LPIPS: 0.3008][Best PSNR: 0.0000][Best Epoch: 0]
save model...
38it [00:08,  4.74it/s]
38it [00:07,  4.77it/s]
Forward: 8.29s
```

Estimated runtime for 300 epochs: 578.4 sec * 300 epochs / (3600 sec / hour) = 48.20 hours

### Batch size of 256

```text
[10240/46839]	[0.0505]	131.9+570.1s
[20480/46839]	[0.0352]	57.9+0.1s
[30720/46839]	[0.0310]	57.6+0.1s
[40960/46839]	[0.0261]	57.6+0.2s
Epoch 1 cost time: 938.8s, lr: 0.000100
save model...
0it [00:00, ?it/s]Testing...
38it [00:14,  2.65it/s]
38it [00:14,  2.64it/s]
38it [00:15,  2.51it/s]
38it [00:14,  2.62it/s]
38it [00:14,  2.63it/s]
38it [00:14,  2.63it/s]
38it [00:14,  2.63it/s]
38it [00:14,  2.63it/s]
[Epoch: 1]
[PSNR: 29.6043][SSIM: 0.7781][LPIPS: 0.3312][Best PSNR: 0.0000][Best Epoch: 0]
save model...
Forward: 14.61s
```

Estimated runtime for 300 epochs: 938.8 sec * 300 epochs / (3600 sec / hour) = 78.23 hours
