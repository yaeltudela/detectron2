#!/usr/bin/env bash

# python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_baseline.yaml"
# python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_DA.yaml"
# python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_wHD.yaml"

python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_baseline.yaml" MODEL.ROI_BOX_HEAD.NAME "SplitFastRCNNConvFCHead" OUTPUT_DIR "results/paper_exps/base_head"
python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_DA.yaml" MODEL.ROI_BOX_HEAD.NAME "SplitFastRCNNConvFCHead" OUTPUT_DIR "results/paper_exps/DA_mods_head"
python tools/train_net.py --config-file "configs/polyp/paper/faster_R_101_FPN_wHD.yaml" MODEL.ROI_BOX_HEAD.NAME "SplitFastRCNNConvFCHead" OUTPUT_DIR "results/paper_exps/wHD_head"


