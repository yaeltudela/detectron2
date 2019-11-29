import os

import cv2
import numpy as np
import pandas as pd

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None):
        self.dataset_name = None
        self.dataset_folder = MetadataCatalog.get(dataset_name).


        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self.sequences, self.gt = self._load_gt()

    def _load_gt(self):
        gt_df = pd.read_csv(os.path.join("datasets", self.dataset_folder, "annots", "gt.csv"))
        sequences = gt_df.sequence.nunique()

        return sequences, gt_df

    def _gt_metrics(self, gt_image):
        im = cv2.imread(gt_image, cv2.IMREAD_GRAYSCALE)
        polyp_in_frame = im.sum() > 0

        return np.array([])

    def _is_polyp_localizated(self, pred, gt):
        x_pred, y_pred, w_pred, h_pred = pred
        x_gt, y_gt, w_gt, h_gt = gt

    def reset(self):
        pass

    def evaluate(self):
        pass

    def process(self, input, output):
        for input, output in zip(input, output):

            pass
            #pred_boxes =

        im_seq = output['image_id'].split("-")[0]

        np.concatenate([self.vid_results[im_seq], self._gt_metrics()])
