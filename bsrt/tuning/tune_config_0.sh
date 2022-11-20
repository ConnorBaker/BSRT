#!/usr/bin/env bash

python -m bsrt.tuning.objective \
  --batch_size 8 \
  --data_dir "$HOME/datasets" \
  --st_checkpoint_dir "$HOME/syne_tune/checkpoints" \
  --limit_train_batches 0.01 \
  --limit_val_batches 0.01 \
  --max_epochs 20 \
  --precision bfloat16 \
  --BSRTParams.attn_drop_rate=0.14110334702534744 \
  --BSRTParams.drop_path_rate=0.4807023766374288 \
  --BSRTParams.drop_rate=0.5298004515803402 \
  --BSRTParams.flow_alignment_groups=8 \
  --BSRTParams.mlp_ratio=2 \
  --BSRTParams.num_features=8 \
  --BSRTParams.qkv_bias=true \
  --optimizer AdamW \
  --AdamWParams.beta_gradient=0.008432847670872564 \
  --AdamWParams.beta_square=0.6196468349554187 \
  --AdamWParams.eps=3.486226119528844e-8 \
  --AdamWParams.lr=0.0008326518015149353 \
  --AdamWParams.weight_decay=0.0000020716174241670848 \
  --scheduler OneCycleLR \
  --OneCycleLRParams.base_momentum=0.7409099040760416 \
  --OneCycleLRParams.div_factor=64.46944947820214 \
  --OneCycleLRParams.final_div_factor=31018.204665802456 \
  --OneCycleLRParams.max_lr=0.06844716252649218 \
  --OneCycleLRParams.max_momentum=0.7824919898193952 \
  --OneCycleLRParams.pct_start=0.22781029711053424 \
  "$@"
