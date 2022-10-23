#!/usr/bin/env bash

set -euo pipefail

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,source="$(pwd)",target=/BSRT --mount type=bind,source=/home/connorbaker/ramdisk/datasets,target=/home/connorbaker/ramdisk/datasets connorbaker01/bsrt:22.09 python -m bsrt.tuning.tuner