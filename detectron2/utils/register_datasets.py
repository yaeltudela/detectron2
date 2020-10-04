import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json

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

}

only_polyp_categories = {
    "Polyp": {
        'id': 1,
        'name': 'Polyp',
        'supercategory': 'polyp',
    },
}

polyp_dataset_categories_polyp = ["Polyp"]

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
        "evaluator_type": "giana"
    },
    "ETIS_LaribPolypDB": {
        "split": "etis.json",
        "categories": ["Polyp","Polyp2"],
        # "categories": ["AD", "NAD"],

        "evaluator_type": "giana"
    }

}


def register_polyp_datasets(only_polyp=False):
    for dataset_name, dataset_data in polyp_datasets.items():
        polyp_cats = only_polyp_categories if only_polyp else polyp_categories
        if "ETIS" in dataset_name:
            polyp_cats = {
                "Polyp": {
                    'id': 1,
                    'name': 'Polyp',
                    'supercategory': 'polyp',
                },
                "Polyp2": {
                    'id': 2,
                    'name': 'Polyp2',
                    'supercategory': 'polyp',
                },

            }

        root_dir = os.path.join("datasets", dataset_name, "images")
        if only_polyp:
            annot_file = os.path.join("datasets", dataset_name, "annotations", dataset_data['split'].replace(".json", "_polyp.json"))
        else:
            annot_file = os.path.join("datasets", dataset_name, "annotations", dataset_data['split'])

        print(annot_file)
        dataset_data['categories'] = polyp_dataset_categories_polyp if only_polyp else dataset_data['categories']
        print(dataset_data['categories'])
        thing_ids = [v["id"] for k, v in polyp_cats.items() if k in dataset_data['categories']]
        # Mapping from the incontiguous COCO category id to an id in [0, 79]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [v["name"] for k, v in polyp_cats.items() if k in dataset_data['categories']]

        metadata = {
            "thing_classes": thing_classes,
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "evaluator_type": dataset_data['evaluator_type'],
            'annot_file': annot_file
        }

        register_coco_instances(name=dataset_name, metadata=metadata, json_file=annot_file, image_root=root_dir)

    print("Polyp datasets registered.")


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, **metadata)
