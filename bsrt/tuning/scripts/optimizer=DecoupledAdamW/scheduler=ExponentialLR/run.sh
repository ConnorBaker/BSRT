#!/usr/bin/env bash

set -euo pipefail

docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --rm connorbaker01/bsrt:22.09 python \
  -m bagua.distributed.run \
  --enable_bagua_net \
  --autotune_level 1 \
  -m bsrt.tuning.tuner \
  --experiment_name "DecoupledAdamWExponentialLR" \
  --optimizer "DecoupledAdamW" \
  --scheduler "ExponentialLR" \
  --precision "bf16" \
  --num_trials 1000 \
  --max_epochs 100 \
  --batch_size 24 \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0 \
  "${@}"
