import datetime
import os

import cv2
import numpy as np
import pandas as pd
from skimage import measure


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def im_info(id, filename, size):
    return {
        "id": id,
        "width": size[1],
        "height": size[0],
        "file_name": filename,
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": "",
    }


def annot_info(id, img_id, cat_id, segm, area, bbox):
    return {
        "id": id,
        "image_id": img_id,
        "category_id": cat_id,
        "segmentation": segm,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }


def gt_to_coco(root_dir, coco_info, categories, split_name):
    gt_file = os.path.join(root_dir, "gt.csv")
    image_dir = os.path.join(root_dir, "images")
    masks_dir = os.path.join(root_dir, "masks")

    if not os.path.exists(gt_file):
        assert FileNotFoundError("gt file")
    df = pd.read_csv(gt_file)
    assert not {"image", "mask", "has_polyp", "classification"}.issubset(
        df.columns), "annot file doesn't have 'image', 'has_polyp', 'classifcation', or 'mask' columns. "

    coco_images = {}
    coco_annots = []

    image_id = 1
    annot_id = 1

    df_images = pd.unique(df.image)
    true_categories = []
    for im in df_images:
        im_annots = df[df.image == im]

        coco_images[im] = im_info(image_id, im, cv2.imread(os.path.join(image_dir, im), 0).shape)
        for idx, row in im_annots.iterrows():

            if row.has_polyp:
                mask = cv2.imread(os.path.join(masks_dir, row['mask']), 0)

                ys, xs = np.where(mask != 0)
                segm = binary_mask_to_polygon((mask != 0).astype('uint8'))
                box = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]
                box = [int(b) for b in box]
                area = float((mask != 0).sum() / (mask.shape[0] * mask.shape[1]))

                # only for debug reasons
                if row['class'] not in categories.keys():
                    print(row)
                # add used categories to dataset
                if categories[row['class']]['name'] not in true_categories:
                    true_categories.append(categories[row['class']]['name'])

                coco_annots.append(annot_info(annot_id, image_id, categories[row['class']]['id'], segm, area, box))
                annot_id += 1
            else:
                pass

        image_id += 1

    coco_images = [im for k, im in coco_images.items()]

    import json
    out = {
        "info": coco_info,
        "licenses": [{
            "id": 1,
            "name": "---",
            "url": "---"
        }],
        "categories": [item for k, item in categories.items() if k in true_categories],
        "images": coco_images,
        "annotations": coco_annots
    }

    output_file = os.path.join(root_dir, "annotations", split_name + ".json")
    with open(output_file, "w") as f:
        json.dump(out, f)


if __name__ == '__main__':
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

    categories_hd = {
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

    categories_polyp = {
        # Default class # not sure about how this works
        "Polyp": {
            'id': 1,
            'name': 'Polyp',
            'supercategory': 'polyp',
        }
    }
    # print("clinic")
    # gt_to_coco("../../datasets/CVC_ClinicDB", coco_info, categories, "clinic")
    # print("colon")
    # gt_to_coco("../../datasets/CVC_ColonDB", coco_info, categories, "colon")
    # print("hd")
    # gt_to_coco("../../datasets/CVC_HDClassif", coco_info, categories, "hdClassif")
    gt_to_coco("../../datasets/CVC_HDClassif_test", coco_info, categories, "hdClassif_test")

    # print("test")
    # gt_to_coco("../../datasets/CVC_VideoClinicDB_test", coco_info, categories, "test")
    # print("train")
    # gt_to_coco("../../datasets/CVC_VideoClinicDB_train", coco_info, categories, "train")
    # print("valid")
    # gt_to_coco("../../datasets/CVC_VideoClinicDB_valid", coco_info, categories, "valid")
    # print("etis")
    # gt_to_coco("../../datasets/ETIS_LaribPolypDB", coco_info, categories_polyp, "etis")
