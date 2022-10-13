#!/usr/bin/env bash

python bsrt/main.py fit --config bsrt/configs/data.yml --config bsrt/configs/model.yml --config bsrt/configs/optimizer.yml --config bsrt/configs/trainer_test.yml "$@"