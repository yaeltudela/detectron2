python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd.yaml"
python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_hd.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/baselines/mask_all_hd"

# python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_base.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/baselines/mask_base"
# python tools/train_net.py --config-file "configs/polyp/baselines/faster_R_50_FPN_post.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/baselines/mask_all"

# python projects/TridentNet/train_net.py --config-file "projects/TridentNet/configs/polyp/trident_base.yaml"
# python projects/TridentNet/train_net.py --config-file "projects/TridentNet/configs/polyp/trident_post.yaml"
# python projects/TridentNet/train_net.py --config-file "projects/TridentNet/configs/polyp/trident_base.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/baselines/trident_mask_base"
# python projects/TridentNet/train_net.py --config-file "projects/TridentNet/configs/polyp/trident_post.yaml" MODEL.MASK_ON True OUTPUT_DIR "results/baselines/trident_mask_base"