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
  --BSRTParams.drop_path_rate=0.7331222695104113 \
  --BSRTParams.drop_rate=0.2110009578648595 \
  --BSRTParams.flow_alignment_groups=4 \
  --BSRTParams.mlp_ratio=32 \
  --BSRTParams.num_features=68 \
  --BSRTParams.qkv_bias=0 \
  --optimizer AdamW \
  --AdamWParams.beta_gradient=0.0011865482765193352 \
  --AdamWParams.beta_square=0.7354839845870553 \
  --AdamWParams.eps=4.4872086216491385e-08 \
  --AdamWParams.lr=8.777414422374876e-06 \
  --AdamWParams.weight_decay=1.1459835260153865e-08 \
  --scheduler OneCycleLR \
  --OneCycleLRParams.base_momentum=0.7855988406789076 \
  --OneCycleLRParams.div_factor=310.71141220241924 \
  --OneCycleLRParams.final_div_factor=55664.694700643195 \
  --OneCycleLRParams.max_lr=0.0556038961971246 \
  --OneCycleLRParams.max_momentum=0.9125376194454561 \
  --OneCycleLRParams.pct_start=0.4499869794240212 \
  "$@"
