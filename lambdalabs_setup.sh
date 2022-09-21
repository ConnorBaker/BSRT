#!/usr/bin/env bash

cd ~
curl micro.mamba.pm/install.sh | bash && source ~/.bashrc

sudo apt update && sudo apt install zstd -y
pip install nvitop --user

mkdir working && cd ./working
git clone https://github.com/ConnorBaker/BSRT

gcloud storage cp gs://bsrt-supplemental/models.tar.zst .
tar -xf models.tar.zst

cd ./BSRT
micromamba create -f ./environment.yml -y && micromamba activate bsrt
cd ./bsrt

python main.py --data_type synthetic --lr 0.0001 --decay 100-200 --model_level S --swinfeature --batch_size 4 --burst_size 14 --patch_size 256 --data_dir ~/working/datasets --models_root ~/working/models --epochs 100 --use_checkpoint