#!/usr/bin/env bash
# python tools/train_net.py --resume --eval-only --config-file "results/results/ablation/hvflips_10rotation_mr/config.yaml" MODEL.WEIGHTS "results/results/ablation/hvflips_10rotation_mr/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3


# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/hd_base" DATASETS.TRAIN "('CVC_HDClassif',)" SOLVER.MAX_ITER 3312 SOLVER.STEPS '(0, 1656, 2650)' DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', )"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.WEIGHTS "results/results/all_exps/quality/hd_base/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/hd_then_video"  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', )"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/video_base" DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', )"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.WEIGHTS "results/results/all_exps/quality/video_base/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/video_then_hd" DATASETS.TRAIN "('CVC_HDClassif',)" SOLVER.MAX_ITER 3312 SOLVER.STEPS '(0, 1656, 2650)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', )"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/concatenated" DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 27887 SOLVER.STEPS '(0, 13943, 22308)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)"

#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/hd_base" DATASETS.TRAIN "('CVC_HDClassif',)" SOLVER.MAX_ITER 3312 SOLVER.STEPS '(0, 1656, 2650)' DATASETS.TEST "('CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test', 'CVC_ColonDB', 'CVC_ClinicDB', )"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.WEIGHTS "results/results/all_exps/quality/hd_base/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/hd_then_video"  DATASETS.TEST "('CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test', 'CVC_ColonDB', 'CVC_ClinicDB', )"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/video_base" DATASETS.TEST "('CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test', 'CVC_ColonDB', 'CVC_ClinicDB', )"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.WEIGHTS "results/results/all_exps/quality/video_base/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/video_then_hd" DATASETS.TRAIN "('CVC_HDClassif',)" SOLVER.MAX_ITER 3312 SOLVER.STEPS '(0, 1656, 2650)'  DATASETS.TEST "('CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test', 'CVC_ColonDB', 'CVC_ClinicDB', )"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.WEIGHTS "results/results/all_exps/quality/concatenated/model_final.pth" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/quality/concatenated" DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 27887 SOLVER.STEPS '(0, 13943, 22308)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)"



# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/giou"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/ciou"
# python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(1.,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/l1_giou"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.ROI_HEADS.WCE_LOSS True TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/wce"

#python tools/train_net.py --resume --eval-only --config-file "results/results/all_exps/losses/wce/config.yaml"  MODEL.ROI_HEADS.WCE_LOSS True  TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/wce" DATASETS.TEST "('CVC_VideoClinicDB_test', )"
#python tools/train_net.py --resume --eval-only --config-file "results/results/all_exps/losses/giou/config.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/giou" DATASETS.TEST "('CVC_VideoClinicDB_test', )"
#python tools/train_net.py --resume --eval-only --config-file "results/results/all_exps/losses/ciou/config.yaml" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/ciou" DATASETS.TEST "('CVC_VideoClinicDB_test', )"
#python tools/train_net.py --resume --eval-only --config-file "results/results/all_exps/losses/l1_giou/config.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(1.,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/l1_giou" DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', )"
#python tools/train_net.py --resume --eval-only --config-file "results/results/all_exps/losses/hl1_giou/config.yaml" MODEL.ROI_BOX_HEAD.USE_GIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/hl1_giou" DATASETS.TEST "('CVC_VideoClinicDB_test', )"
#python tools/train_net.py --resume --eval-only --config-file "results/results/all_exps/losses/hl1_ciou/config.yaml" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/losses/hl1_ciou" DATASETS.TEST "('CVC_VideoClinicDB_test', )"

#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.IOU_THRESHOLDS "[0.4, 0.7]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/rpn/ious4_7"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.IOU_THRESHOLDS "[0.4, 0.6]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/rpn/ious4_6"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.IOU_THRESHOLDS "[0.3, 0.6]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/rpn/ious3_6"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS "[[0.25, 0.5, 1.0, 1.5, 2.0]]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/rpn/aspect_ratios"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.POSITIVE_FRACTION 0.25 TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/rpn/pos_frac"

#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.POSITIVE_FRACTION 0.25 MODEL.RPN.IOU_THRESHOLDS "[0.4, 0.7]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/rpn/ious4_7_pos"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.POSITIVE_FRACTION 0.25 MODEL.RPN.IOU_THRESHOLDS "[0.4, 0.7]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/combination/da_loss_rpn" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)'
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RPN.POSITIVE_FRACTION 0.25 MODEL.RPN.IOU_THRESHOLDS "[0.4, 0.7]" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/combination/da_loss_rpn_sam3" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"


#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/maskiou/mask"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.MASKIOU_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/maskiou/maskiou"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/maskiou/mask_hd" DATASETS.TRAIN "('CVC_HDClassif',)" SOLVER.MAX_ITER 3312 SOLVER.STEPS '(0, 1656, 2650)' DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.MASKIOU_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/maskiou/maskiou_hd" DATASETS.TRAIN "('CVC_HDClassif',)" SOLVER.MAX_ITER 3312 SOLVER.STEPS '(0, 1656, 2650)' DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)"

#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam3"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 5 TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam5"
#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 7 TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam7"

#python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam3" DATASETS.TEST "('CVC_VideoClinicDB_test',)"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 5 MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam5" DATASETS.TEST "('CVC_VideoClinicDB_test',)"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 7 MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam7" DATASETS.TEST "('CVC_VideoClinicDB_test',)"





#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam3"

#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/sam/sam7_hd"



#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' MODEL.RPN.POSITIVE_FRACTION 0.25 OUTPUT_DIR "results/results/all_exps/combi/losses_rpn"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 MODEL.MASKIOU_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/combi/maskiou_sam3"


#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' MODEL.RPN.POSITIVE_FRACTION 0.25 OUTPUT_DIR "results/results/all_exps/combi/losses_rpn"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 MODEL.MASKIOU_ON True MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" TEST.DETECTIONS_PER_IMAGE 3 OUTPUT_DIR "results/results/all_exps/combi/maskiou_sam3"


python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' OUTPUT_DIR "results/results/all_exps/combi/losses_101_2x" SOLVER.MAX_ITER 49150 SOLVER.STEPS "(0, 24575, 39320)" DATASETS.TEST "('CVC_VideoClinicDB_test',)"
#python tools/train_net.py --resume --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl" OUTPUT_DIR "results/results/all_exps/combi/mask_101_sam3_2x" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 SOLVER.MAX_ITER 49150 SOLVER.STEPS "(0, 24575, 39320)"



#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' OUTPUT_DIR "results/results/all_exps/combi/mask_faster_changes"
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"  OUTPUT_DIR "results/results/all_exps/combi/concatenated101" DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 27887 SOLVER.STEPS '(0, 13943, 22308)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)'
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" TEST.DETECTIONS_PER_IMAGE 3 MODEL.RESNETS.DEPTH 101 OUTPUT_DIR "results/results/all_exps/combi/concatenated_mask101" DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 27887 SOLVER.STEPS '(0, 13943, 22308)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
#
#python tools/train_net.py --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' OUTPUT_DIR "results/results/all_exps/combi/mask_faster_changes101"


 python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 55774 SOLVER.STEPS '(0, 27887, 44619)'  DATASETS.TEST "('CVC_VideoClinicDB_test',)" OUTPUT_DIR "results/results/all_exps/final/f101_losses_concat"

#python tools/train_net.py  --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.WEIGHTS "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 TEST.DETECTIONS_PER_IMAGE 3 MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' OUTPUT_DIR "results/results/all_exps/final/m101_losses_sam"


# python tools/train_net.py  --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" TEST.DETECTIONS_PER_IMAGE 3 DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 55774 SOLVER.STEPS '(0, 27887, 44619)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)" MODEL.ROI_BOX_HEAD.USE_CIOU True MODEL.ROI_HEADS.LOSS_WEIGHTS '(0.5,1.,1.)' MODEL.ROI_HEADS.BOX_HEAD_TYPE 'double' OUTPUT_DIR "results/results/all_exps/final/f101_losses_concat_double_2x"
 python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.MASK_ON True MODEL.ROI_MASK_HEAD.SAM_ON True MODEL.ROI_MASK_HEAD.SAM_K 3 SOLVER.MAX_ITER 49150 SOLVER.STEPS "(0, 24575, 39320)" OUTPUT_DIR "results/results/all_exps/final/m101_sam_2x" DATASETS.TEST "('CVC_VideoClinicDB_test',)"
 python tools/train_net.py --resume --eval-only --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.MASK_ON True DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 55774 SOLVER.STEPS '(0, 27887, 44619)'  OUTPUT_DIR "results/results/all_exps/final/m101_concat_sam_2x" DATASETS.TEST "('CVC_VideoClinicDB_test', )"
# python tools/train_net.py  --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.MASK_ON True DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 55774 SOLVER.STEPS '(0, 27887, 44619)' MODEL.MASKIOU_ON True DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)" OUTPUT_DIR "results/results/all_exps/final/m101_concat_maskiou_sam_2x"
# python tools/train_net.py  --config-file "configs/polyp/da_ablation/faster_R_50_FPN_10rotation_flips.yaml" MODEL.RESNETS.DEPTH 101 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl" TEST.DETECTIONS_PER_IMAGE 3 MODEL.MASK_ON True DATASETS.TRAIN "('CVC_HDClassif', 'CVC_VideoClinicDB_train')" SOLVER.MAX_ITER 55774 SOLVER.STEPS '(0, 27887, 44619)'  DATASETS.TEST "('CVC_ColonDB', 'CVC_ClinicDB', 'CVC_VideoClinicDB_valid', 'CVC_VideoClinicDB_test',)" MODEL.ROI_HEADS.BOX_HEAD_TYPE 'double' OUTPUT_DIR "results/results/all_exps/final/m101_concat_maskiou_double_2x"
