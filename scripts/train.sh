#!/bin/bash

accelerate launch \
    --config_file "scripts/configs/zero3.yaml" \
    src/main.py \
    --config "configs/gravex_openfake.yaml" \