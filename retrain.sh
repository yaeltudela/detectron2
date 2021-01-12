# python tools/train_net.py --config-file /media/devsodin/WORK/TFM/results/baselines/faster/faster_base_hd/config.yaml OUTPUT_DIR results/video_then_hd MODEL.WEIGHTS /media/devsodin/WORK/TFM/results/baselines/faster/faster_base_hd/model_final.pth
# python tools/train_net.py --config-file /media/devsodin/WORK/TFM/results/baselines/faster/faster_base/config.yaml OUTPUT_DIR results/hd_then_video MODEL.WEIGHTS /media/devsodin/WORK/TFM/results/baselines/faster/faster_base/model_final.pth

# python tools/train_net.py --config-file /media/devsodin/WORK/TFM/results/baselines/mask/mask_base_hd/config.yaml OUTPUT_DIR results/m_video_then_hd MODEL.WEIGHTS /media/devsodin/WORK/TFM/results/baselines/mask/mask_base_hd/model_final.pth
python tools/train_net.py --config-file /media/devsodin/WORK/TFM/results/baselines/mask/mask_base/config.yaml OUTPUT_DIR results/m_hd_then_video MODEL.WEIGHTS /media/devsodin/WORK/TFM/results/baselines/mask/mask_base/model_final.pth
