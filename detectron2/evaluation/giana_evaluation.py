import os

import pandas as pd

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None):
        self.dataset_name = MetadataCatalog.get(dataset_name).name.split("__")[0]
        self.classes = MetadataCatalog.get(dataset_name).get("thing_dataset_id_to_contiguous_id")
        self.dataset_folder = os.path.join("datasets", self.dataset_name)
        self.output_folder = output_dir

        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self.gt = self._load_gt()

        self.results = pd.DataFrame(columns=["image", "detected", "localized", "classified", "score"])

    def _load_gt(self):
        return pd.read_csv(os.path.join(self.dataset_folder, "gt.csv"))

    def _is_polyp_localizated(self, pred, gt):
        return False

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
        print(self.results)
        self.results[['sequence', 'frame']] = self.results.image.str.split("-", expand=True)
        sequences = pd.unique(self.results.sequence)
        dets = []
        locs = []
        avg_df_detection = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        avg_df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
        for sequence in sequences:
            df_detection = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
            df_localization = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN"])
            filtered = self.results[self.results.sequence == sequence]
            print(filtered)
            for threshold in self.thresholds:
                th_cond = (filtered.score >= threshold) | (filtered.score == -1)
                thresholded = filtered[th_cond]
                under_threshold = filtered[~th_cond]

                det = thresholded.drop_duplicates(subset="image", keep="first").detected.value_counts()
                under_det = under_threshold.drop_duplicates(subset="image", keep="first").detected.value_counts()
                print(det)
                det_tp = det.TP if "TP" in det.keys() else 0
                det_fp = det.FP if "FP" in det.keys() else 0
                det_tn = (det.TN if "TN" in det.keys() else 0) + (under_det.FP if "FP" in under_threshold.keys() else 0)
                det_fn = (det.FN if "FN" in det.keys() else 0) + (under_det.TP if "TP" in under_threshold.keys() else 0)
                self._add_row(df_detection, [threshold, det_tp, det_fp, det_tn, det_fn])

                loc = thresholded.localized.value_counts()
                under_loc = under_threshold.localized.value_counts()

                loc_tp = loc.TP if "TP" in loc.keys() else 0
                loc_fp = loc.FP if "FP" in loc.keys() else 0
                loc_tn = (loc.TN if "TN" in loc.keys() else 0) + (under_det.FP if "FP" in under_loc.keys() else 0)
                loc_fn = (loc.FN if "FN" in loc.keys() else 0) + (under_det.TP if "TP" in under_loc.keys() else 0)
                self._add_row(df_localization, [threshold, loc_tp, loc_fp, loc_tn, loc_fn])

            df_detection.to_csv(self.output_folder + "/d{}.csv".format(sequence), index=False)
            df_localization.to_csv(self.output_folder + "/l{}.csv".format(sequence), index=False)
            dets.append(df_detection)
            locs.append(df_localization)

        for det, loc in zip(dets, locs):
            avg_df_detection = pd.concat([avg_df_detection, det])
            avg_df_localization = pd.concat([avg_df_localization, loc])

        avg_df_detection.groupby("threshold").count().to_csv("d_avg.csv")
        avg_df_localization.groupby("threshold").count().to_csv("l_avg.csv")

        self.results.to_csv(self.output_folder + "/results.csv", index=False)

    def _add_row(self, df, row):
        df.loc[len(df)] = row
        df.index += 1
        df.reset_index(inplace=True, drop=True)

    def process(self, input, output):

        for instance, output in zip(input, output):
            im_name = os.path.basename(instance['file_name'])

            fields = output["instances"].get_fields()
            pred_boxes = fields['pred_boxes']
            scores = fields['scores'].cpu().numpy()
            pred_class = fields['pred_classes']

            gt_has_polyp, gt_classifcations, gt_centers = self.get_gt_info(im_name)
            pred_has_polyp = len(pred_boxes) > 0
            detection_response = self._is_polyp_detected(pred_has_polyp, gt_has_polyp)

            # Model has predictions for the frame
            if len(pred_boxes) > 0:
                if gt_has_polyp:
                    # FPS and TPS
                    # Find matches from all predictions with groundTruth
                    for box, score, classif in zip(pred_boxes, scores, pred_class):
                        to_check = len(gt_centers)
                        checked = 0
                        for gt_class, center in zip(gt_classifcations, gt_centers):
                            # if pred box cross gt center; is valid localization
                            if box[0] < center[0] < box[2] and box[1] < center[1] < box[3]:
                                gt_centers.remove(center)
                                gt_classifcations.remove(gt_class)
                                localization_response = "TP"
                                classification_response = classif
                                break
                            else:
                                checked += 1
                                if checked == to_check:
                                    localization_response = "FP"
                                    classification_response = "non-eval"
                                    break
                        row = [im_name, detection_response, localization_response, classification_response, score]
                        self._add_row(self.results, row)

                else:
                    localization_response = "FP"
                    for box, score, classif in zip(pred_boxes, scores, pred_class):
                        row = [im_name, detection_response, localization_response, "non-eval", score]
                        self._add_row(self.results, row)
            # Model has no preds for the frame
            else:
                if gt_has_polyp:
                    localization_response = "FN"
                    for gt_class, center in zip(gt_classifcations, gt_centers):
                        row = [im_name, detection_response, localization_response, "non-eval", -1]
                        self._add_row(self.results, row)
                else:
                    localization_response = "TN"
                    row = [im_name, detection_response, localization_response, "non-eval", -1]
                    self._add_row(self.results, row)

    def get_gt_info(self, im_name):
        classifcations = []
        centers = []
        has_polyp = False
        image_gt = self.gt[self.gt.image == im_name]
        for row in image_gt.iterrows():
            idx, row = row
            has_polyp = row.has_polyp
            if has_polyp:
                classifcations.append(row['class'])
                centers.append((row.center_x, row.center_y))
        return has_polyp, classifcations, centers
