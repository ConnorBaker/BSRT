#!/usr/bin/env bash

echo "Train the small model on synthetic data"

python main.py --n_GPUs 8 --data_type synthetic --print_every 40 --lr 0.0001 --decay 100-200 --save bsrt_tiny_synthetic --model BSRT --fp16 --model_level S --swinfeature --batch_size 32 --burst_size 14 --patch_size 256
# python main.py --n_GPUs 8 --data_type synthetic --print_every 40 --lr 0.0001 --decay 100-200 --save bsrt_large_synthetic --model BSRT --fp16 --model_level L --swinfeature --batch_size 16 --burst_size 14 --patch_size 256

echo "Test the small model on synthetic data"

python test_burst.py --n_GPUs 1 --data_type synthetic --model BSRT --model_level S --fp16 --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_tiny_synthetic/bsrt_best_epoch.pth --root /data/dataset/ntire21/burstsr/synthetic
# python test_burst.py --n_GPUs 1 --data_type synthetic --model BSRT --model_level L --fp16 --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_large_synthetic/bsrt_best_epoch.pth --root /data/dataset/ntire21/burstsr/synthetic

echo "Train the small model on real data using the previous model"

python main.py --n_GPUs 8 --data_type real --print_every 20 --lr 0.00004 --decay 40-80 --save bsrt_tiny_real --model BSRT --fp16 --model_level S --swinfeature --batch_size 8 --burst_size 14 --patch_size 80 --pre_train ../train_log/bsrt/real_models/bsrt_tiny_synthetic/bsrt_best_epoch.pth
# python main.py --n_GPUs 8 --data_type real --print_every 20 --lr 0.00004 --decay 40-80 --save bsrt_large --model BSRT --fp16 --model_level L --swinfeature --batch_size 8 --burst_size 14 --patch_size 48 --pre_train  ../train_log/bsrt/real_models/bsrt_large_synthetic/bsrt_best_epoch.pth

echo "Test the small model on real data"

python test.py --n_GPUs 1 --data_type real --model BSRT --model_level S --swinfeature --batch_size 1 --burst_size 14 --patch_size 80 --pre_train ../train_log/bsrt/real_models/bsrt_tiny_real/bsrt_best_epoch.pth --root /data/dataset/ntire21/burstsr/real
# python test.py --n_GPUs 1 --data_type real --model BSRT --model_level L --swinfeature --batch_size 1 --burst_size 14 --patch_size 80 --pre_train ../train_log/bsrt/real_models/bsrt_large_real/bsrt_realworld.pth --root /data/dataset/ntire21/burstsr/real