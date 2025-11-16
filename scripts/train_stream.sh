#!/bin/bash

accelerate launch \
    --config_file "scripts/configs/zero3.yaml" \
    src/main_streaming.py \
    --config "configs/streaming_gravex_openfake.yaml" \