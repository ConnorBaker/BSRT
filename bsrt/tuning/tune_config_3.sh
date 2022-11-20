#!/usr/bin/env bash

python -m bsrt.tuning.objective \
  --batch_size 8 \
  --data_dir "$HOME/datasets" \
  --st_checkpoint_dir "$HOME/syne_tune/checkpoints" \
  --limit_train_batches 0.01 \
  --limit_val_batches 0.01 \
  --max_epochs 20 \
  --precision bfloat16 \
  --BSRTParams.attn_drop_rate=0.429204053285285 \
  --BSRTParams.drop_path_rate=0.6605375367004296 \
  --BSRTParams.drop_rate=0.05707869750230388 \
  --BSRTParams.flow_alignment_groups=4 \
  --BSRTParams.mlp_ratio=8 \
  --BSRTParams.num_features=48 \
  --BSRTParams.qkv_bias=True \
  --optimizer AdamW \
  --AdamWParams.beta_gradient=0.7354839845870553 \
  --AdamWParams.beta_square=0.04022294569134715 \
  --AdamWParams.eps=4.4872086216491385e-08 \
  --AdamWParams.lr=0.10755294118707269 \
  --AdamWParams.weight_decay=0.0006514953713444319 \
  --scheduler OneCycleLR \
  --OneCycleLRParams.base_momentum=0.7855988406789076 \
  --OneCycleLRParams.div_factor=660.8042978534147 \
  --OneCycleLRParams.final_div_factor=93301.16151899766 \
  --OneCycleLRParams.max_lr=0.006860952442922341 \
  --OneCycleLRParams.max_momentum=0.7947055796269176 \
  --OneCycleLRParams.pct_start=0.11568312769975009 \
  "$@"
