#!/usr/bin/env bash

python -m bsrt.tuning.objective \
  --batch_size 8 \
  --data_dir "$HOME/datasets" \
  --st_checkpoint_dir "$HOME/syne_tune/checkpoints" \
  --limit_train_batches 0.01 \
  --limit_val_batches 0.01 \
  --max_epochs 20 \
  --precision bfloat16 \
  --BSRTParams.attn_drop_rate=0.891624751134527 \
  --BSRTParams.drop_path_rate=0.210139234438378 \
  --BSRTParams.drop_rate=0.2110009578648595 \
  --BSRTParams.flow_alignment_groups=4 \
  --BSRTParams.mlp_ratio=8 \
  --BSRTParams.num_features=48 \
  --BSRTParams.qkv_bias=0 \
  --optimizer AdamW \
  --AdamWParams.beta_gradient=0.9307517935966583 \
  --AdamWParams.beta_squre=0.0691477496102281 \
  --AdamWParams.eps=5.085593968632704e-09 \
  --AdamWParams.lr=0.0006968262915767198 \
  --AdamWParams.weight_decay=1.1459835260153865e-08 \
  --scheduler OneCycleLR \
  --OneCycleLRParams.base_momentum=0.7855988406789076 \
  --OneCycleLRParams.div_factor=310.71141220241924 \
  --OneCycleLRParams.final_div_factor=37481.33256138692 \
  --OneCycleLRParams.max_lr=0.05471692447460814 \
  --OneCycleLRParams.max_momentum=0.9298882022730891 \
  --OneCycleLRParams.pct_start=0.35535010389548793 \
  "$@"
