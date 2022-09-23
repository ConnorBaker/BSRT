#!/usr/bin/env bash

cd ~
curl micro.mamba.pm/install.sh | bash && source ~/.bashrc

sudo apt update && sudo apt install zstd -y
pip install nvitop --user

mkdir working && cd ./working
git clone https://github.com/ConnorBaker/BSRT

cd ./BSRT
micromamba create -f ./conda/environment.yml -y && micromamba activate bsrt
cd ./bsrt

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" python -m bagua.distributed.launch --report_metrics --enable_bagua_net --nproc_per_node 1 --autotune_level 1 main.py --data_type synthetic --lr 0.0001 --decay 100-200 --model_level S --swinfeature --batch_size 16 --burst_size 14 --patch_size 256 --data_dir ~/working/datasets --use_checkpoint --max_epochs 10 --limit_train_batches 0.001 --loss L1 --amp_backend native --precision bf16