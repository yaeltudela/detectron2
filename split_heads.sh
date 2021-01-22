#!/usr/bin/env bash
 python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_101_FPN_baseline.yaml" OUTPUT_DIR "results/split_head/mask_base" MODEL.ROI_HEADS.BOX_HEAD_TYPE "double" MODEL.MASK_ON True
 python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_combi.yaml" OUTPUT_DIR "results/split_head/mask_da" MODEL.ROI_HEADS.BOX_HEAD_TYPE "double" MODEL.MASK_ON True
 python tools/train_net.py --config-file "configs/polyp/dummy_rpn.yaml" OUTPUT_DIR "results/split_head/modrpn_mask_da" MODEL.ROI_HEADS.BOX_HEAD_TYPE "double" MODEL.MASK_ON True
#python tools/train_net.py --config-file "configs/polyp/dummy_rpn.yaml" OUTPUT_DIR "results/split_head/modrpn_mask_da_polyp" MODEL.ROI_HEADS.BOX_HEAD_TYPE "double" MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.MASK_ON True
