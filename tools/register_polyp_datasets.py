import os
from glob import glob

import cv2
import pandas as pd

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode


def get_polyp_dataset(ims_path, annot_file, ims_extension):
    assert os.path.exists(ims_path)

    ims_files = sorted(glob(os.path.join(ims_path, "*.{}".format(ims_extension))))

    polyp_dataset = []
    assert os.path.exists(annot_file)
    annots = pd.read_csv(annot_file)

    for i, im in enumerate(ims_files):
        h, w = cv2.imread(im).shape[:2]
        im_name = os.path.basename(im)

        im_annots_df = annots[annots.image == im_name]

        im_annots = []

        for i, annot in im_annots_df.iterrows():
            if annot['x'] == -1:
                continue
            else:

                mask = cv2.imread(im.replace("images/", "masks/").replace(annot['image'], annot['mask']), 0)
                contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                poly = []

                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        poly.append(contour)
                if len(poly) == 0:
                    continue

                cat_id, box = annot['class'], [annot['x'], annot['y'], annot['w'], annot['h']]

                annot = {
                    "category_id": cat_id,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "bbox": box,
                    "segmentation": poly
                }
                im_annots.append(annot)

        im_data = {
            "file_name": im,
            "image_id": i,
            "height": h,
            "width": w,
            "annotations": im_annots
        }
        polyp_dataset.append(im_data)

    return polyp_dataset


def register_polyp_dataset(dataset_name, ims_path, annot_file, ims_extension):
        def register(): return get_polyp_dataset(ims_path, annot_file, ims_extension)

        DatasetCatalog.register(dataset_name, register)
        MetadataCatalog.get(dataset_name).set(thing_classes=[k for k, v in {"NA": 1, "AD": 2}])

        convert_to_coco_json(dataset_name, '/home/devsodin/{}.json'.format(dataset_name), allow_cached=False)

        print("{} registered".format(dataset_name))

if __name__ == '__main__':
    # dataset_names = ['CVC_HDClassif', "CVC_VideoClinicDB_train", "CVC_VideoClinicDB_valid", "CVC_VideoClinicDB_test"]
    # ims_paths = ['../datasets/CVC_HDClassif/images/', "../datasets/CVC_VideoClinicDB_train/images/", "../datasets/CVC_VideoClinicDB_valid/images/",
    #              "../datasets/CVC_VideoClinicDB_test/images/"]
    ims_extensions = ["png", "png", "png"]
    dataset_names = ["ColonDB", 'ClinicDB']
    ims_paths = ['../datasets/CVC_ColonDB/images/', '../datasets/CVC_ClinicDB/images/']
    annot_files = [ims_path.replace("images/", "gt2.csv") for ims_path in ims_paths]
    ims_extensions = ['bmp', 'bmp']

    for dataset_name, ims_path, annot_file, ims_extension in zip(dataset_names, ims_paths, annot_files, ims_extensions):
        register_polyp_dataset(dataset_name, ims_path, annot_file, ims_extension)
