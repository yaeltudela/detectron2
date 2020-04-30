import os

import pandas as pd

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures.boxes import BoxMode


class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None, old_metric=False):
        self.dataset_name = MetadataCatalog.get(dataset_name).name.split("__")[0]
        self.classes_id = MetadataCatalog.get(dataset_name).get("thing_dataset_id_to_contiguous_id")
        self.class_id_name = {v: k for k, v in
                              zip(MetadataCatalog.get(dataset_name).get("thing_classes"), self.classes_id.values())}
        self.dataset_folder = os.path.join("datasets", self.dataset_name)
        self.output_folder = os.path.join(output_dir, "giana")
        self.detection_folder = os.path.join(output_dir, "detection")
        self.localization_folder = os.path.join(output_dir, "localization")
        self.classification_folder = os.path.join(output_dir, "classification")
        self.old_metric = old_metric

        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self.gt = self._load_gt()

        self.results = pd.DataFrame(columns=["image", "detected", "localized", "classified", "score", "pred_box"])

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

    def _load_gt(self):
        return pd.read_csv(os.path.join(self.dataset_folder, "gt.csv"))

    def _is_polyp_detected(self, pred, gt):
        if pred:
            if gt:
                return "TP"
            else:
                return "FP"
        else:
            if gt:
                return "FN"
            else:
                return "TN"

    def reset(self):
        pass

    def evaluate(self):
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

            df_detection.to_csv(
                os.path.join(self.detection_folder, "d{}{}.csv".format(sequence, "_old" if self.old_metric else "")),
                index=False)
            df_localization.to_csv(
                os.path.join(self.localization_folder, "l{}{}.csv".format(sequence, "_old" if self.old_metric else "")),
                index=False)
            df_classification.to_csv(os.path.join(self.classification_folder,
                                                  "c{}{}.csv".format(sequence, "_old" if self.old_metric else "")),
                                     index=False)
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

        self.results.to_csv(os.path.join(self.output_folder, "results{}.csv".format("_old" if self.old_metric else "")),
                            index=False)

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
        df.to_csv(os.path.join(save_folder, "avg{}.csv".format("_old" if self.old_metric else "")), index=False)

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

    def _add_row(self, df, row):
        df.loc[len(df)] = row
        df.index += 1
        df.reset_index(inplace=True, drop=True)

    def process(self, input, output):

        for instance, output in zip(input, output):
            already_localized = False
            im_name = os.path.basename(instance['file_name'])

            fields = output["instances"].get_fields()
            pred_boxes = fields['pred_boxes']
            scores = fields['scores'].cpu().numpy()
            pred_class = fields['pred_classes']

            gt_has_polyp, gt_classifcations, gt_centers, gt_boxes = self.get_gt_info(im_name)
            pred_has_polyp = len(pred_boxes) > 0
            detection_response = self._is_polyp_detected(pred_has_polyp, gt_has_polyp)

            # Model has predictions for the frame
            if len(pred_boxes) > 0:
                if gt_has_polyp:
                    # FPS and TPS
                    # Find matches from all predictions with groundTruth
                    for pred_box, pred_score, pred_classif in zip(pred_boxes, scores, pred_class):
                        pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
                        to_check = len(gt_centers)
                        checked = 0
                        for gt_classif, gt_center, gt_box in zip(gt_classifcations, gt_centers, gt_boxes):
                            # if pred box cross gt center; is valid localization
                            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
                            gt_cx, gt_cy = gt_center

                            if self.old_metric:
                                # predicted center inside box
                                pred_cx, pred_cy = (pred_x1 + (pred_x2 - pred_x1) / 2), (pred_y1 + (pred_y2 - pred_y1) / 2)
                                eval_condition = (gt_x1 < pred_cx < gt_x2) and (gt_y1 < pred_cy < gt_y2)
                            else:
                                # predicted box contains gt center
                                eval_condition = (pred_x1 < gt_cx < pred_x2) and (pred_y1 < gt_cy < pred_y2)

                            if eval_condition:
                                gt_centers.remove(gt_center)
                                gt_classifcations.remove(gt_classif)
                                gt_boxes.remove(gt_box)
                                localization_response = "TP" if not already_localized else "FP"
                                already_localized = True
                                classification_response = self._is_polyp_classified(
                                    self.class_id_name[pred_classif.item()], gt_classif)

                                break
                            else:
                                # check if match any gt center
                                checked += 1
                                if checked == to_check:
                                    localization_response = "FP"
                                    classification_response = "non-eval"
                                    break
                        row = [im_name, detection_response, localization_response, classification_response, pred_score, pred_box]
                        self._add_row(self.results, row)

                else:
                    localization_response = "FP"
                    for pred_box, pred_score, pred_classif in zip(pred_boxes, scores, pred_class):
                        row = [im_name, detection_response, localization_response, "non-eval", pred_score, pred_box]
                        self._add_row(self.results, row)
            # Model has no preds for the frame
            else:
                if gt_has_polyp:
                    localization_response = "FN"
                    for gt_classif, gt_center in zip(gt_classifcations, gt_centers):
                        row = [im_name, detection_response, localization_response, "non-eval", -1, "NA"]
                        self._add_row(self.results, row)
                else:
                    localization_response = "TN"
                    row = [im_name, detection_response, localization_response, "non-eval", -1, "NA"]
                    self._add_row(self.results, row)

    def get_gt_info(self, im_name):
        classifcations = []
        centers = []
        boxes = []
        has_polyp = False
        image_gt = self.gt[self.gt.image == im_name]
        for row in image_gt.iterrows():
            idx, row = row
            has_polyp = row.has_polyp
            if has_polyp:
                classifcations.append(row['class'])
                centers.append((row.center_y, row.center_x))
                boxes.append([row.y_min, row.x_min, row.y_max, row.x_max])
        return has_polyp, classifcations, centers, boxes

    @staticmethod
    def _is_polyp_classified(pred, gt):
        if pred == gt:
            if pred == 'AD':
                return "TP"
            else:
                return "TN"
        else:
            if pred == 'AD':
                return "FP"
            else:
                return "FN"


def offline_evaluation(dataset_name, output_dir, results_file):
    evaluator = GianaEvaulator(dataset_name, output_dir)
    evaluator.results = pd.read_csv(results_file)
    evaluator.evaluate()


if __name__ == '__main__':
    from argparse import ArgumentParser
    from detectron2.utils.register_datasets import register_polyp_datasets

    register_polyp_datasets()
    ap = ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--file")
    opts = ap.parse_args()

    offline_evaluation(opts.dataset, opts.output, opts.file)
