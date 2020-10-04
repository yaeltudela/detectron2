#!/usr/bin/env bash

# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_base.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_constrast.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_upscale2_0.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_25rotation.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_upscale.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_5rotation.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_vflip.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_cropping.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_upscale1_5.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_flips.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_brightness.yaml"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_hflip.yaml"




python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/5rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/5rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/5rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/10rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/10rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/10rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/25rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/25rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/25rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/blur_hvflips_10rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/blur_hvflips_10rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/blur_hvflips_10rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/brightness/config.yaml" MODEL.WEIGHTS "results/results/ablation/brightness/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/brightness/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/color_hvflips_10rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/color_hvflips_10rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/color_hvflips_10rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/contrast/config.yaml" MODEL.WEIGHTS "results/results/ablation/contrast/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/contrast/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/cropping/config.yaml" MODEL.WEIGHTS "results/results/ablation/cropping/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/cropping/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/crops_hvflips_10rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/crops_hvflips_10rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/crops_hvflips_10rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/hflip/config.yaml" MODEL.WEIGHTS "results/results/ablation/hflip/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/hflip/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/hvflips/config.yaml" MODEL.WEIGHTS "results/results/ablation/hvflips/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/hvflips/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/hvflips_10rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/hvflips_10rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/hvflips_10rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/hvflips_10rotation_mr/config.yaml" MODEL.WEIGHTS "results/results/ablation/hvflips_10rotation_mr/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/hvflips_10rotation_mr/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/min_328/config.yaml" MODEL.WEIGHTS "results/results/ablation/min_328/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/min_328/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/min_432/config.yaml" MODEL.WEIGHTS "results/results/ablation/min_432/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/min_432/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/min_576/config.yaml" MODEL.WEIGHTS "results/results/ablation/min_576/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/min_576/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/up15_hvflips_10rotation/config.yaml" MODEL.WEIGHTS "results/results/ablation/up15_hvflips_10rotation/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/up15_hvflips_10rotation/"
python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/vflip/config.yaml" MODEL.WEIGHTS "results/results/ablation/vflip/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/ablation/vflip/"

