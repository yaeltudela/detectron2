import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets import register_coco


def register_polyp_datasets():

    polyp_categories = {
        "AD": {
            'id': 1,
            'name': 'AD',
            'supercategory': 'polyp',
        },
        "NAD": {
            'id': 2,
            'name': 'NAD',
            'supercategory': 'polyp',
        },
        "ASS": {
            'id': 2,
            'name': 'NAD',
            'supercategory': 'polyp',
        },
        "HP": {
            'id': 2,
            'name': 'NAD',
            'supercategory': 'polyp',
        },
        # Default class # not sure about how this works
        "Polyp": {
            'id': 3,
            'name': 'Polyp',
            'supercategory': 'polyp',
        }

    }
    polyp_datasets = {
        "CVC_VideoClinicDB_train": {
            "split": "train.json",
            "categories": ["AD", "NAD"],
            "evaluator_type": "giana"
        },
        "CVC_VideoClinicDB_valid": {
            "split": "valid.json",
            "categories": ["AD", "NAD"],
            "evaluator_type": "giana"
        },
        "CVC_VideoClinicDB_test": {
            "split": "test.json",
            "categories": ["AD", "NAD"],
            "evaluator_type": "giana"
        },
        "CVC_ClinicDB": {
            "split": "clinic.json",
            "categories": ["AD", "NAD"],
            "evaluator_type": "giana"
        },
        "CVC_ColonDB": {
            "split": "colon.json",
            "categories": ["AD", "NAD"],
            "evaluator_type": "giana"
        },
        "CVC_HDClassif": {
            "split": "hdClassif.json",
            "categories": ["AD", "NAD"],
            # "categories": ["AD", "NAD", "Polyp"],
            "evaluator_type": "giana"
        },
        "ETIS_LaribPolypDB": {
            "split": "etis.json",
            "categories": ["Polyp"],
            "evaluator_type": "giana"
        }

    }

    for dataset_name, dataset_data in polyp_datasets.items():
        root_dir = os.path.join("datasets", dataset_name, "images")
        annot_file = os.path.join("datasets", dataset_name, "annotations", dataset_data['split'])

        thing_ids = [v["id"] for k, v in polyp_categories.items() if k in dataset_data['categories']]
        # Mapping from the incontiguous COCO category id to an id in [0, 79]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [v["name"] for k, v  in polyp_categories.items() if k in dataset_data['categories']]

        metadata = {
            "thing_classes": thing_classes,
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id
        }

        register_coco.register_coco_instances(name=dataset_name, metadata=metadata, json_file=annot_file, image_root=root_dir)
    print("Polyp datasets registered.")
