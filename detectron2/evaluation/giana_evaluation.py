from detectron2.evaluation.evaluator import DatasetEvaluator
from glob import glob
from detectron2.data import MetadataCatalog

import os

class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None):
        self.dataset_name = None
        self.dataset_folder = MetadataCatalog.get(dataset_name).image_root

        if thresholds is None:
            self.thresholds = [x / 10 for x in range(10)]
        else:
            self.thresholds = thresholds

        self.sequences = self._get_sequences()

    def _get_sequences(self):
        sequences = glob(os.path.join("datasets",self.dataset_folder, "masks", "*tif"))
        sequences = set([x.split("-")[0] for x in sequences])

        return list(sequences)

    def reset(self):
        pass

    def evaluate(self):
        pass

    def process(self, input, output):


        for input, output in zip(input, output):
            print(input)
            print(output)