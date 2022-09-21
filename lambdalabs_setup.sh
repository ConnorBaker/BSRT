#!/usr/bin/env bash

cd ~
curl micro.mamba.pm/install.sh | bash && source ~/.bashrc

# For Debian 10
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && sudo apt install ./cuda-keyring_1.0-1_all.deb

# For Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && sudo apt install ./cuda-keyring_1.0-1_all.deb

sudo apt update && sudo apt install libnccl2 zstd -y
pip install nvitop --user

mkdir working && cd ./working
git clone https://github.com/ConnorBaker/BSRT

cd ./BSRT
micromamba create -f ./conda/environment.yml -y && micromamba activate bsrt
cd ./bsrt

python main.py --data_type synthetic --lr 0.0001 --decay 100-200 --model_level S --swinfeature --batch_size 4 --burst_size 14 --patch_size 256 --data_dir ~/working/datasets --epochs 100 --use_checkpoint