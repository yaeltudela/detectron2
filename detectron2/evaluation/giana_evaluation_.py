import os

import cv2
import pandas as pd
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, Instances


class GianaEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, output_dir, thresholds=None, old_metric=False, metric_type=None, save_patches=False) -> None:
        self.dataset_name = MetadataCatalog.get(dataset_name).name.split("__")[0]
        self.classes_id = MetadataCatalog.get(dataset_name).get("thing_dataset_id_to_contiguous_id")
        self.annot_file = MetadataCatalog.get(dataset_name).get('annot_file')

        self.class_id_name = {v: k for k, v in
                              zip(MetadataCatalog.get(dataset_name).get("thing_classes"), self.classes_id.values())}
        self.dataset_folder = os.path.join("datasets", self.dataset_name)
        self.output_folder = os.path.join(output_dir, "giana")
        self.detection_folder = os.path.join(output_dir, "detection")
        self.localization_folder = os.path.join(output_dir, "localization")
        self.classification_folder = os.path.join(output_dir, "classification")
        self.old_metric = old_metric
        self.eval_function = self._set_eval_function()
        self.save_patches = save_patches
        # self.save_patches = True

        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self.coco_gt = COCO(self.annot_file)

        self.results = pd.DataFrame(
            columns=["image", "detected", "localized", "classified", "pred_class", "gt_class",
                     "score", "pred_box"])
        self._partial_results = []
        self.make_dirs()

    def make_dirs(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)
        if not os.path.exists(self.localization_folder):
            os.makedirs(self.localization_folder)
        if not os.path.exists(self.classification_folder):
            os.makedirs(self.classification_folder)

    def reset(self):
        self.results = pd.DataFrame(
            columns=["image", "detected", "localized", "classified", "pred_class", "gt_class",
                     "score", "pred_box"])
        self._partial_results = []

    def process(self, input, output):
        for instance, output in zip(input, output):
            im_name = os.path.basename(instance['file_name'])
            im_id = instance['image_id']

            fields = output["instances"].get_fields()
            pred_boxes = fields['pred_boxes']

            pred_scores = fields['scores'].cpu().numpy()
            pred_classes = fields['pred_classes']

            gt_anns = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=im_id))
            gt_boxes = [ann['bbox'] for ann in gt_anns]
            gt_classes = [ann['category_id'] for ann in gt_anns]

            if gt_anns:
                if pred_boxes.tensor.size(0) > 0:
                    # both
                    matchable_gt = list(range(len(gt_anns)))
                    matchable_pred = list(range(pred_boxes.tensor.size(0)))

                    for pred_idx, (pred_bbox, pred_score, pred_class) in enumerate(
                            zip(pred_boxes, pred_scores, pred_classes)):
                        for gt_idx, (gt_bbox, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                            # pred_bbox --> XYXY; gt_bbox --> XYWH
                            gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]] # XYXY
                            if self.eval_function(pred_bbox, gt_bbox):
                                if gt_idx in matchable_gt and pred_idx in matchable_pred:
                                    # TP
                                    matchable_pred.remove(pred_idx)
                                    matchable_gt.remove(gt_idx)
                                    eval_classif = self._is_polyp_classified(pred_class, gt_class)
                                    self._partial_results.append([im_name, "TP", "TP", eval_classif, pred_class.item(), gt_class, pred_score.item(), pred_bbox.cpu().numpy().tolist(), gt_bbox])
                                    self.maybe_save_patch(im_name, pred_bbox, type="TP/{}".format(gt_class), margin=10)

                    for unmatched_gt in matchable_gt:
                        self._partial_results.append([im_name, "FN", "FN", "NA", -1, gt_classes[unmatched_gt], -1, -1, -1])

                    for unmatched_pred in matchable_pred:
                        self._partial_results.append(
                            [im_name, "FP", "FP", "NA", pred_classes[unmatched_pred].item(), -1, pred_scores[unmatched_pred].item(), pred_boxes.tensor[unmatched_pred].cpu().numpy().tolist(), -1])
                        self.maybe_save_patch(im_name, pred_boxes.tensor[unmatched_pred], type='FP', margin=10)

                else:
                    # FN de todos los gt anns
                    for idx, (gt_bbox, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                        self._partial_results.append([im_name, "FN", "FN", "NA", -1, gt_class, -1, -1, gt_bbox])
            else:
                if pred_boxes.tensor.size(0) > 0:
                    # FP de todos los pred anns
                    for pred_bbox, pred_score, pred_class in zip(pred_boxes, pred_scores, pred_classes):
                        self._partial_results.append([im_name, "FP", "FP", "NA", -1, -1, -1, -1, -1])
                else:
                    # TN x 1
                    self._partial_results.append([im_name, "TN", "TN", "NA", -1, -1, -1, -1, -1])

    def _add_row(self, df, row):
        df.loc[len(df)] = row
        df.index += 1
        df.reset_index(inplace=True, drop=True)

    def evaluate(self):
        self.results = pd.DataFrame(self._partial_results,
                                    columns=["image", "detected", "localized", "classified", "pred_class", "gt_class",
                                             "score", "pred_box", "gt_box"])
        self.results[['sequence', 'frame']] = self.results.image.str.split("-", expand=True)

        sequences = pd.unique(self.results.sequence)
        dets = []
        locs = []
        classifs = []
        avg_df_detection = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        avg_df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        avg_df_classification = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        for sequence in sequences:
            df_detection = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])
            df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT", "mIoU"])
            df_classification = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
            filtered = self.results[self.results.sequence == sequence]
            filtered_det = self.results[self.results.sequence == sequence].drop_duplicates(subset="image")
            for threshold in self.thresholds:
                th_cond = (filtered.score >= threshold) | (filtered.score == -1)
                over_threshold = filtered[th_cond]
                under_threshold = filtered[~th_cond]

                miou, ious = self.compute_miou(over_threshold.pred_box, over_threshold.gt_box)

                over_threshold_det = filtered_det[th_cond].drop_duplicates(subset="image")
                under_threshold_det = filtered_det[~th_cond].drop_duplicates(subset="image")

                det = over_threshold_det.detected.value_counts()
                under_det = under_threshold_det.detected.value_counts()
                det_tp = det.TP if "TP" in det.keys() else 0
                det_fp = det.FP if "FP" in det.keys() else 0
                det_tn = (det.TN if "TN" in det.keys() else 0) + (under_det.FP if "FP" in under_det.keys() else 0)
                det_fn = (det.FN if "FN" in det.keys() else 0) + (under_det.TP if "TP" in under_det.keys() else 0)
                first_polyp = over_threshold_det[over_threshold_det.detected == "FN"].frame.apply(
                    lambda x: int(x.split(".")[0])).min()
                first_det_polyp = over_threshold_det[over_threshold_det.detected == "TP"].frame.apply(
                    lambda x: int(x.split(".")[0]))
                first_det_polyp = first_det_polyp[first_det_polyp >= first_polyp].min()
                det_rt = first_det_polyp - first_polyp
                self._add_row(df_detection, [threshold, det_tp, det_fp, det_tn, det_fn, det_rt])

                loc = over_threshold.localized.value_counts()
                under_loc = under_threshold.localized.value_counts()

                loc_tp = loc.TP if "TP" in loc.keys() else 0
                loc_fp = loc.FP if "FP" in loc.keys() else 0
                loc_tn = (loc.TN if "TN" in loc.keys() else 0) + (under_loc.FP if "FP" in under_loc.keys() else 0)
                loc_fn = (loc.FN if "FN" in loc.keys() else 0) + (under_loc.TP if "TP" in under_loc.keys() else 0)
                first_polyp = over_threshold[over_threshold.localized == "FN"].frame.apply(
                    lambda x: int(x.split(".")[0])).min()
                first_loc_polyp = over_threshold[over_threshold.localized == "TP"].frame.apply(
                    lambda x: int(x.split(".")[0]))
                first_loc_polyp = first_loc_polyp[first_loc_polyp >= first_polyp].min()
                loc_rt = first_loc_polyp - first_polyp
                self._add_row(df_localization, [threshold, loc_tp, loc_fp, loc_tn, loc_fn, loc_rt, miou])

                clasif = over_threshold[over_threshold.localized == "TP"].classified.value_counts()

                class_tp = clasif.TP if "TP" in clasif.keys() else 0
                class_fp = clasif.FP if "FP" in clasif.keys() else 0
                class_tn = clasif.TN if "TN" in clasif.keys() else 0
                class_fn = clasif.FN if "FN" in clasif.keys() else 0
                self._add_row(df_classification, [threshold, class_tp, class_fp, class_tn, class_fn])

            df_detection.to_csv(os.path.join(self.detection_folder, "d{}.csv".format(sequence)), index=False)
            df_localization.to_csv(os.path.join(self.localization_folder, "l{}.csv".format(sequence)), index=False)
            df_classification.to_csv(os.path.join(self.classification_folder, "c{}.csv".format(sequence)), index=False)
            dets.append(df_detection)
            locs.append(df_localization)
            classifs.append(df_classification)
        print("computing Averages and aggregation metrics")
        for det, loc, classif in zip(dets, locs, classifs):
            avg_df_detection = pd.concat([avg_df_detection, det], ignore_index=True, sort=False)
            avg_df_localization = pd.concat([avg_df_localization, loc], ignore_index=True, sort=False)
            avg_df_classification = pd.concat([avg_df_classification, classif], ignore_index=True, sort=False)

        self.compute_average_metrics(avg_df_detection, len(sequences), self.detection_folder)
        self.compute_average_metrics(avg_df_localization, len(sequences), self.localization_folder)
        self.compute_average_metrics(avg_df_classification, len(sequences), self.classification_folder)

        self.results.to_csv(os.path.join(self.output_folder, "results.csv"), index=False)

    def _set_eval_function(self):
        if self.old_metric:
            return self.old_eval
        else:
            return self.new_eval

    def compute_average_metrics(self, df, n_sequences, save_folder):
        df = df.groupby("threshold")
        if 'mIoU' in df.sum().columns:
            std_miou = df.std().mIoU

        if "RT" in df.sum().columns:
            stdRT = df.std().RT
            mRT = df.sum().RT.apply(lambda x: round(x / n_sequences, 2))
            stdRT = stdRT.round(2)

        df = df.sum()

        if 'mIoU' in df.columns:
            df['mIoU'] = df.mIoU / n_sequences
            df['std_mIoU'] = std_miou

        if 'RT' in df.columns:
            df['mRT'] = mRT
            df['stdRT'] = stdRT

        df = self._compute_aggregation_metrics(df)
        df.to_csv(os.path.join(save_folder, "avg.csv"), index=False)

    def _compute_aggregation_metrics(self, df):
        tp = df.TP
        fp = df.FP
        tn = df.TN
        fn = df.FN

        acc = (tp + tn) / (tp + fp + tn + fn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1core = 2 * pre * rec / (pre + rec)

        df['accuracy'] = acc.round(4)
        df["precision"] = pre.round(4)
        df["recall"] = rec.round(4)
        df["f1score"] = f1core.round(4)

        return df

    def old_eval(self, pred_box, gt_box):
        """

        :param pred_box: box xyxy
        :param gt_box: box xyxy
        :return:
        """
        pred_cx = pred_box[0] + ((pred_box[2] - pred_box[0]) / 2)
        pred_cy = pred_box[1] + ((pred_box[3] - pred_box[1]) / 2)

        return (gt_box[0] <= pred_cx <= gt_box[0] + gt_box[2]) and (gt_box[1] <= pred_cy <= gt_box[1] + gt_box[3])

    def new_eval(self, pred_box, gt_box):

        gt_cx = gt_box[0] + ((gt_box[2] - gt_box[0]) / 2)
        gt_cy = gt_box[1] + ((gt_box[3] - gt_box[1]) / 2)

        return (pred_box[0] <= gt_cx <= pred_box[2]) and (pred_box[1] <= gt_cy <= pred_box[3])

    def _is_polyp_classified(self, pred, gt):
        if pred == gt:
            if pred == 2:
                return "TP"
            elif pred == 1:
                return "TN"
            else:
                raise Exception("Pred {} - GT {}".format(pred, gt))
        else:
            if pred == 1:
                return "FP"
            elif pred == 2:
                return "FN"
            else:
                raise Exception("Pred {} - GT {}".format(pred, gt))

    def maybe_save_patch(self, im_name, pred_bbox, type, margin=0):
        # pred box XYXYX
        if self.save_patches:
            if not os.path.exists(os.path.join(self.output_folder, "patches")):
                os.makedirs(os.path.join(self.output_folder, "patches"), exist_ok=True)
                os.makedirs(os.path.join(self.output_folder, "patches", "TP", "0"), exist_ok=True)
                os.makedirs(os.path.join(self.output_folder, "patches", "TP", "1"), exist_ok=True)
                os.makedirs(os.path.join(self.output_folder, "patches", "FP"), exist_ok=True)

            im_file = os.path.join(self.dataset_folder, "images", im_name)
            im = cv2.imread(im_file)

    def compute_miou(self, pred_boxes, gt_boxes):
        ious = []

        for pred, gt in zip(pred_boxes, gt_boxes):
            if pred == -1 and gt == -1:
                continue
            ious.append(compute_iou(pred, gt))

        miou = (sum(ious) + 1e-6) / (len(ious) + 1e-6)

        return miou, ious

def compute_iou(pred_box, gt_box):
    if pred_box == -1 or gt_box == -1:
        return 0
    x_left = max(pred_box[0], gt_box[0])
    y_top = max(pred_box[1], gt_box[1])
    x_right = min(pred_box[2], gt_box[2])
    y_bottom = min(pred_box[3], gt_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    bb2_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area + 1e-5)

    return iou


def offline_evaluation(dataset_name, output_dir, results_file):
    evaluator = GianaEvaluator(dataset_name, output_dir)
    evaluator.results = pd.read_csv(results_file)
    evaluator.evaluate()


def offline_eval_from_coco_res(dataset_name, output_dir, coco_res):
    import numpy as np
    import torch
    evaluator = GianaEvaluator(dataset_name, output_dir)

    for det_img in coco_res.imgs.values():
        im_id = det_img['id']
        file_name = det_img['file_name']
        image_size = (det_img['height'], det_img['width'])

        det_anns = coco_res.loadAnns(coco_res.getAnnIds(imgIds=im_id))

        # XYWH json format --> Boxes XYXY_ABS
        boxes = []
        scores = []
        pred_classes = []
        for det_annot in det_anns:
            boxes.append(det_annot['bbox'])
            scores.append(det_annot['score'])
            pred_classes.append(det_annot['category_id'])

        boxes = np.array(boxes)
        if boxes.size == 0:
            boxes = Boxes(torch.tensor([]).view(-1, 4))
        else:
            boxes[:, 2:] += boxes[:, :2]
            boxes = Boxes(torch.tensor(boxes).view(-1, 4))
        scores = torch.tensor(scores).view(-1, 1)
        pred_classes = np.array(pred_classes).reshape(-1, 1)

        det_input = {
            'image_id': im_id,
            'file_name': file_name
        }
        det_out = {
                'pred_boxes': boxes, # Boxes
                'scores': scores, # np array
                'pred_classes': pred_classes # tensor
        }

        print(det_input)
        print(det_out)

        det_out = Instances(image_size, **det_out)

        evaluator.process([det_input], [{'instances': det_out}])

    evaluator.evaluate()



if __name__ == '__main__':
    from argparse import ArgumentParser
    from detectron2.utils.register_datasets import register_polyp_datasets

    register_polyp_datasets()

    gt = COCO("/home/devsodin/PycharmProjects/detectron2/datasets/CVC_VideoClinicDB_test/annotations/test.json")
    offline_eval_from_coco_res("CVC_VideoClinicDB_test", "/home/devsodin/test", gt.loadRes("/home/devsodin/PycharmProjects/detectron2/results/baselines/faster_all/inference/coco_instances_results.json"))
