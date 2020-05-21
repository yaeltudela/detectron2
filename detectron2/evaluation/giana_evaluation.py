import os

import pandas as pd
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes


class GianaEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, output_dir, thresholds=None, old_metric=False) -> None:
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
                                    self._partial_results.append(
                                        [im_name, "TP", "TP", eval_classif, pred_class, gt_class, pred_score, pred_bbox])

                    for unmatched_gt in matchable_gt:
                        self._partial_results.append([im_name, "FN", "FN", "NA", -1, gt_classes[unmatched_gt], -1, -1])

                    for unmatched_pred in matchable_pred:
                        self._partial_results.append(
                            [im_name, "FP", "FP", "NA", pred_classes[unmatched_pred], -1, pred_scores[unmatched_pred],
                             pred_boxes[unmatched_pred]])

                else:
                    # FN de todos los gt anns
                    for idx, (gt_bbox, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                        self._partial_results.append([im_name, "FN", "FN", "NA", -1, gt_class, -1, -1])
            else:
                if pred_boxes.tensor.size(0) > 0:
                    # FP de todos los pred anns
                    for pred_bbox, pred_score, pred_class in zip(pred_boxes, pred_scores, pred_classes):
                        self._partial_results.append([im_name, "FP", "FP", "NA", -1, -1, -1, -1])
                else:
                    # TN x 1
                    self._partial_results.append([im_name, "TN", "TN", "NA", -1, -1, -1, -1])

    def _add_row(self, df, row):
        df.loc[len(df)] = row
        df.index += 1
        df.reset_index(inplace=True, drop=True)

    def evaluate(self):
        self.results = pd.DataFrame(self._partial_results,
                                    columns=["image", "detected", "localized", "classified", "pred_class", "gt_class",
                                             "score", "pred_box"])
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
            df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])
            df_classification = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
            filtered = self.results[self.results.sequence == sequence]
            filtered_det = self.results[self.results.sequence == sequence].drop_duplicates(subset="image")
            for threshold in self.thresholds:
                th_cond = (filtered.score >= threshold) | (filtered.score == -1)
                over_threshold = filtered[th_cond]
                under_threshold = filtered[~th_cond]

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
                self._add_row(df_localization, [threshold, loc_tp, loc_fp, loc_tn, loc_fn, loc_rt])

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

    def compute_average_metrics(self, df, sequences, save_folder):
        df = df.groupby("threshold")
        if "RT" in df.sum().columns:
            stdRT = df.std().RT
            df = df.sum()
            df['mRT'] = df.RT.apply(lambda x: round(x / sequences, 2))
            df['stdRT'] = stdRT.round(2)
        else:
            df = df.sum()
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
            if pred == 1:
                return "TP"
            elif pred == 0:
                return "TN"
            else:
                raise NotImplemented
        else:
            if pred == 0:
                return "FP"
            elif pred == 1:
                return "FN"
            else:
                raise NotImplemented


def offline_evaluation(dataset_name, output_dir, results_file):
    evaluator = GianaEvaluator(dataset_name, output_dir)
    evaluator.results = pd.read_csv(results_file)
    evaluator.evaluate()


def offline_eval_from_coco_res(coco_gt, coco_res):
    results = []

    for k, gt_metadata in coco_gt.imgs.items():
        image_id = gt_metadata['id']
        filename = gt_metadata['file_name']
        gt_anns_boxes = [a['bbox'] for a in coco_gt.loadAnns(coco_gt.getAnnIds(image_id))]
        det_anns_boxes = [(a['bbox'], a['score']) for a in coco_res.loadAnns(coco_res.getAnnIds(image_id))]

        positive_frame = len(gt_anns_boxes) > 0
        positive_det = len(det_anns_boxes) > 0

        if positive_frame:
            if positive_det:
                for det_box, det_score in det_anns_boxes:
                    if gt_anns_boxes:
                        det_cx = det_box[0] + det_box[2]
                        det_cy = det_box[1] + det_box[3]

                        for gt_box in gt_anns_boxes:
                            gt_box[2] = gt_box[0] + gt_box[2]
                            gt_box[3] = gt_box[1] + gt_box[3]

                            if gt_box[0] <= det_cx <= gt_box[2] and gt_box[1] <= det_cy <= gt_box[3]:
                                results.append([filename, "TP", "TP", "NA", det_score, det_box])
                                gt_anns_boxes.remove(gt_box)
                            else:
                                results.append([filename, "TP" if positive_frame else "TN",
                                                "FP" if positive_frame else "TN", "NA", det_score, det_box])
                    else:
                        results.append(
                            [filename, "TP" if positive_frame else "TN", "FP" if positive_frame else "TN", "NA",
                             det_score, det_box])
            else:
                for gt_box in gt_anns_boxes:
                    results.append([filename, "FN", "FN", "NA", 1., "-"])
        else:
            if positive_det:
                for det_box, det_score in det_anns_boxes:
                    results.append([filename, "FP", "FP", "NA", det_score, det_box])
            else:
                results.append([filename, "TN", "TN", "NA", 1., "-"])

    pd.DataFrame(results, columns=["image", "detected", "localized", "classified", "score", "pred_box"])


if __name__ == '__main__':
    from argparse import ArgumentParser
    from detectron2.utils.register_datasets import register_polyp_datasets

    register_polyp_datasets()

    gt = COCO("/home/devsodin/PycharmProjects/detectron2/datasets/CVC_VideoClinicDB_test/annotations/test.json")
    offline_eval_from_coco_res(gt, gt.loadRes("/home/devsodin/PycharmProjects/TernausNet_v2/dummy.json"))

    ap = ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--file")
    opts = ap.parse_args()

    offline_evaluation(opts.dataset, opts.output, opts.file)
