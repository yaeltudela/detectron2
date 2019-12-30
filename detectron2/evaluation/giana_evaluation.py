import os

import cv2
import numpy as np
import pandas as pd

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None):
        self.dataset_name = MetadataCatalog.get(dataset_name).name.split("__")[0]
        self.dataset_folder = os.path.join("datasets", self.dataset_name)
        self.results = pd.DataFrame(columns=['seq', "frame", "TP", "FP", "TN", "FN"])

        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self.sequences, self.gt = self._load_gt()

    def _load_gt(self):
        gt_df = pd.read_csv(os.path.join(self.dataset_folder, "gt.csv"))
        sequences = gt_df.sequence.nunique()

        return sequences, gt_df

    def _gt_metrics(self, gt_image):
        im = cv2.imread(gt_image, cv2.IMREAD_GRAYSCALE)
        polyp_in_frame = im.sum() > 0

        return np.array([])

    def _is_polyp_localizated(self, pred, gt):
        preds_centers = pred.get_centers().cpu().numpy()
        x_min, y_min, x_max, y_max, cx, cy = gt

        # old metric condition
        for center in preds_centers:
            if x_max >= center[0] >= x_min:
                if y_max >= center[1] >= y_min:
                    return True

        return False

    def _is_polyp_classificated(self, pred, gt):
        pass

    def _is_polyp_detected(self, pred, gt):
        pass

    def reset(self):
        pass

    def evaluate(self):
        print(self.results)

    def _add_row(self, row):
        self.results.loc[len(self.results)] = row
        self.results.index += 1
        self.results.reset_index(inplace=True, drop=True)

    def process(self, input, output):

        for input, output in zip(input, output):
            im_name = input['file_name'].split("/")[-1]
            fields = output["instances"].get_fields()

            image_gt = self.gt[self.gt.image == im_name]

            pred_boxes = fields['pred_boxes']
            scores = fields['scores']
            pred_class = fields['pred_classes']

            for row in image_gt.iterrows():

                idx, row = row

                if row.has_polyp:
                    # TP y FN
                    if self._is_polyp_localizated(pred_boxes,
                                               row[['x_min', 'y_min', 'x_max', 'y_max', 'center_x', 'center_y']]):
                        self._add_row([row.sequence, row.frame, True, False, False, False])
                    else:
                        self._add_row([row.sequence, row.frame, False, False, False, True])
                else:
                    if pred_boxes.tensor.shape[0] > 0:
                        self._add_row([row.sequence, row.frame, False, True, False, False])
                    else:
                        self._add_row([row.sequence, row.frame, False, False, True, False])
