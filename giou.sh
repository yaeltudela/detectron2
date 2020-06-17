#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_box/faster_hd"
#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd_all.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_box/faster_hd_all"
#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd.yaml" MODEL.RPN.USE_GIOU True OUTPUT_DIR "results/giou_rpn/faster_hd"
#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd_all.yaml" MODEL.RPN.USE_GIOU True OUTPUT_DIR "results/giou_rpn/faster_hd_all"
#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd.yaml" MODEL.RPN.USE_GIOU True MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_both/faster_hd"
#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd_all.yaml" MODEL.RPN.USE_GIOU True MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_both/faster_hd_all"

python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_base.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_box/faster_v"
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_post.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_box/faster_v_all"
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_base.yaml" MODEL.RPN.USE_GIOU True OUTPUT_DIR "results/giou_rpn/faster_v"
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_post.yaml" MODEL.RPN.USE_GIOU True OUTPUT_DIR "results/giou_rpn/faster_v_all"
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_base.yaml" MODEL.RPN.USE_GIOU True MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_both/faster_v"
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_post.yaml" MODEL.RPN.USE_GIOU True MODEL.ROI_BOX_HEAD.USE_GIOU True OUTPUT_DIR "results/giou_both/faster_v_all"


