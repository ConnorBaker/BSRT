#!/usr/bin/env bash

set -euxo pipefail

# if $HOME/ramdisk doesn't exist, create it
if [ ! -d "$HOME/ramdisk" ]; then
  mkdir "$HOME/ramdisk"
  sudo mount -t tmpfs tmpfs "$HOME/ramdisk"
  sudo chown $USER:$USER -R "$HOME/ramdisk"
  mkdir "$HOME/ramdisk/datasets"
fi

docker run \
  --gpus all \
  --mount type=bind,source="$HOME/ramdisk/datasets",target=/datasets \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --rm connorbaker01/bsrt:main \
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
  "${@}"
