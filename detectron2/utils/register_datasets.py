import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json


def register_polyp_datasets():
    polyp_datasets = {
        "CVC-classification__train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        "cvc-colondb-300__train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        "cvc-colondb-612__train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        "cvcvideoclinicdbtest__test": {
            "split": "test.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana",
            "gt_file": "gt.csv"
        },
        "CVC-VideoClinicDBtrain_valid__train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana",

        },
        "CVC-VideoClinicDBtrain_valid__valid": {
            "split": "valid.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        # "ETIS-LaribPolypDB_train": {
        #     "split": "train.json",
        #     "thing_classes": ["AD", "NA"],
        #     "evaluator_type": "giana"
        # },
    }

    for dataset_name, dataset_data in polyp_datasets.items():
        annot_file = os.path.join("datasets", dataset_name.split("__")[0], "annotations",
                                  dataset_data['split'])
        root_dir = os.path.join("datasets", dataset_name.split("__")[0], "images")
        metadata = {
            "thing_classes": dataset_data['thing_classes'],
            "thing_dataset_id_to_contiguous_id": {i+1: i for i, k in enumerate(dataset_data['thing_classes'])}
        }

        DatasetCatalog.register(dataset_name, lambda: load_coco_json(annot_file, root_dir, dataset_name))

        MetadataCatalog.get(dataset_name).set(
            json_file=annot_file, image_root=root_dir, evaluator_type="giana", **metadata
        )
