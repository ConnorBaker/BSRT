#!/usr/bin/env bash

cd ~
mkdir micromamba
sudo mount -t tmpfs tmpfs ./micromamba
sudo chown -R ubuntu:ubuntu ./micromamba
curl micro.mamba.pm/install.sh | bash

sudo apt update && sudo apt upgrade -y && sudo apt install zstd -y
pip install nvitop --user
gcloud init --no-browser

mkdir working && sudo mount -t tmpfs tmpfs ./working && sudo chown -R ubuntu:ubuntu ./working

cd ./working
git clone https://github.com/ConnorBaker/BSRT

gcloud storage cp gs://bsrt-supplemental/{models,datasets}.tar.zst .
tar -xf models.tar.zst && tar -xf datasets.tar.zst

cd ./BSRT
micromamba create -f ./environment.yml -y
micromamba activate bsrt
cd ./code/synthetic/bsrt

python main.py --n_GPUs 8 --print_every 40 --lr 0.0001 --decay 100-200 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 64 --burst_size 14 --patch_size 256 --root ~/working/datasets/zurich-raw-to-rgb  --val_root ~/working/datasets/SyntheticBurstVal --models_root ~/working/models --epochs 100 --use_checkpoint --save_models