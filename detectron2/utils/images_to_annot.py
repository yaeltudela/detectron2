import datetime

import cv2
import numpy as np
import pandas as pd
from pycocotools import coco


def gen_csv(images_dir, images_extension, masks_dir, output_file, histologies_correspondences):
    images = glob(images_dir + "*." + images_extension)

    data = pd.DataFrame(columns=["image", "mask", "class", "box", "segm"])
    for image in images:
        image = image.split("/")[-1].split(".")[0]
        masks = glob(masks_dir + image + "*")

        seq = image.split("-")[0]
        histology = histologies_correspondences[seq]

        if not masks:
            data.loc[len(data)] = [image, "", "", "", ""]

        for mask in masks:
            im_mask = cv2.imread(mask, 0)

            res, labeled = cv2.connectedComponents(im_mask)

            # TODO filter small components

            boxes = []
            for i in range(1, res):
                ys, xs = np.where(labeled == i)

                segm = coco.maskUtils.encode(np.asfortranarray((labeled == i).astype("uint8")))
                boxes.append([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()])
            for box in boxes:
                data.loc[len(data)] = [image, mask, histology, box, segm['counts']]

    data['image'] = data['image'].map(lambda x: x + ".png")
    data['mask'] = data['mask'].map(lambda x: x.split("/")[-1])
    data.sort_values(["image"], inplace=True)
    data.to_csv(output_file, index=False, sep=",")


def im_info(id, filename):



    return {
        "id": id,
        "width"
        : int,
        "height"
        : int,
        "file_name"
        : str,
        "license"
        : int,
        "flickr_url"
        : str,
        "coco_url"
        : str,
        "date_captured"
        : datetime,
    }


def csv_to_coco(df, coco_info, coco_cateogires, output_file):
    images = []

    for row in df.iterrows()

    coco_output = {
        "info": coco_info,
        "licenses": [{
            "id": 1,
            "name": "---",
            "url": "---"
        }],
        "categories": coco_cateogires,
        "images": [],
        "annotations": []
    }


if __name__ == '__main__':
    from glob import glob

    train_val_histos = {
        "001": "NA",
        "002": "NA",
        "003": "AD",
        "004": "AD",
        "005": "AD",
        "006": "AD",
        "007": "NA",
        "008": "NA",
        "009": "AD",
        "010": "AD",
        "011": "AD",
        "012": "NA",
        "013": "AD",
        "014": "AD",
        "015": "AD",
        "016": "AD",
        "017": "NA",
        "018": "AD",
    }
    test_histos = {
        "001": "AD",
        "002": "NA",
        "003": "AD",
        "004": "AD",
        "005": "AD",
        "006": "AD",
        "007": "AD",
        "008": "AD",
        "009": "AD",
        "010": "AD",
        "011": "AD",
        "012": "AD",
        "013": "AD",
        "014": "AD",
        "015": "AD",
        "016": "NA",
        "017": "AD",
        "018": "AD",
    }

    coco_info = {
        "description": "CVC-clinic",
        "url": "",
        "version": "1.0",
        "year": 2019,
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    categories = {
        "AD": {
            'id': 1,
            'name': 'AD',
            'supercategory': 'polyp',
        },
        "NA": {
            'id': 2,
            'name': 'NA',
            'supercategory': 'polyp',
        }
    }

    # gen_csv("/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/images_train/", "png",
    #         "/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/masks/", "train.csv",
    #         train_val_histos)
    # gen_csv("/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/images_val/", "png",
    #         "/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/masks/", "valid.csv",
    #         train_val_histos)
    # gen_csv("/home/yael/PycharmProjects/detectron2/datasets/cvcvideoclinicdbtest/images/", "png",
    #         "/home/yael/PycharmProjects/detectron2/datasets/cvcvideoclinicdbtest/masks/", "test.csv",
    #         test_histos)
