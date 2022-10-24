#!/usr/bin/env bash

set -euo pipefail

docker run --gpus all \
  --mount type=bind,source="$(pwd)",target=/BSRT \
  --mount type=bind,source=/home/connorbaker/ramdisk/datasets,target=/home/connorbaker/ramdisk/datasets \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  connorbaker01/bsrt:22.09 python \
  -m bagua.distributed.run \
  --enable_bagua_net \
  --autotune_level 1 \
  -m bsrt.tuning.tuner \
  --experiment_name "model_with_adam_plateau" \
  --optimizer "DecoupledAdamW" \
  --scheduler "ReduceLROnPlateau" \
  --precision "bf16" \
  --num_trials 1 \
  --max_epochs 10 \
  --batch_size 16 \
  --limit_train_batches 0.01 \
  --limit_val_batches 0.01 \
  "${@}"
