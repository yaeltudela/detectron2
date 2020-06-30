#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd.yaml" OUTPUT_DIR "results/dml/faster_hd" MODEL.ROI_HEADS.DML_HEAD True
#python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd_all.yaml" OUTPUT_DIR "results/dml/faster_hd_all" MODEL.ROI_HEADS.DML_HEAD True
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_base.yaml"  OUTPUT_DIR "results/dml/faster_vid" MODEL.ROI_HEADS.DML_HEAD True
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_post.yaml" OUTPUT_DIR "results/dml/faster_vid_all" MODEL.ROI_HEADS.DML_HEAD True
