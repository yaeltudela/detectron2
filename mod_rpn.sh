#!/usr/bin/env bash
python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_101_FPN_baseline.yaml"
python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_101_FPN_baseline.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/mod_rpn/mask/video_base" MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_50_FPN_all.yaml"
python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_50_FPN_all.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/mod_rpn/mask/video_da" MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"

#python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_50_FPN_hd.yaml"
#python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_50_FPN_hd_all.yaml"
#python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_50_FPN_hd.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/mod_rpn/mask/hd_base" MODEL.WEIGHTS "detectron2:/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
#python tools/train_net.py --config-file "configs/polyp/mod_rpn/faster_R_50_FPN_hd_all.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/mod_rpn/mask/hd_da" MODEL.WEIGHTS "detectron2:/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
