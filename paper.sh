#!/usr/bin/env bash

# python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_baseline.yaml"
# python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_DA.yaml"
python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_wHD.yaml"
