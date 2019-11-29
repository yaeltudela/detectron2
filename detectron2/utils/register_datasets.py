import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json


def register_polyp_datasets():
    polyp_datasets = {
        "CVC-classification_train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        "cvc-colondb-300_train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        "cvc-colondb-612_train": {
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana"
        },
        "cvcvideoclinicdbtest_test": {
            "dataset_folder": "cvcvideoclinicdbtest",
            "split": "test.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana",
            "gt_file": "gt.csv"
        },
        "CVC-VideoClinicDBtrain_valid_train": {
            "dataset_folder": "CVC-VideoClinicDBtrain_valid",
            "split": "train.json",
            "thing_classes": ["AD", "NA"],
            "evaluator_type": "giana",

        },
        "CVC-VideoClinicDBtrain_valid_valid": {
            "dataset_folder": "CVC-VideoClinicDBtrain_valid",
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
        annot_file = os.path.join("datasets", "_".join(dataset_name.split("_")[:-1]), "annotations",
                                  dataset_data['split'])
        root_dir = os.path.join("datasets", "_".join(dataset_name.split("_")[:-1]), "images")
        metadata = {
            "thing_classes": dataset_data['thing_classes'],
            "thing_dataset_id_to_contiguous_id": {i+1: i for i, k in enumerate(dataset_data['thing_classes'])}
        }

        DatasetCatalog.register(dataset_name, lambda: load_coco_json(annot_file, root_dir, dataset_name))

        MetadataCatalog.get(dataset_name).set(
            json_file=annot_file, image_root=root_dir, evaluator_type="giana", **metadata
        )
