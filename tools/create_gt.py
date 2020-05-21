import os
from glob import glob

import cv2
import numpy as np
import pandas as pd


def get_valid_masks(image, masks):
    im = os.path.basename(image).split(".")[0]

    def filter_masks(mask):
        return True if im in mask else False

    return filter(filter_masks, masks)


def calc_gt(dataset_name, images_folder, ims_ext, masks_folder, masks_ext, check_histology):
    gt = []
    ims = glob(os.path.join(images_folder, f"*.{ims_ext}"))
    masks = glob(os.path.join(masks_folder, f"*.{masks_ext}"))

    for image in ims:
        # print(image)
        im_name = os.path.basename(image)

        im_masks = get_valid_masks(image, masks)
        if not im_masks:
            gt.append([im_name, "", "NA", -1, -1, -1, -1])
        else:
            for im_mask in im_masks:
                classif = check_histology(image)
                mask_name = os.path.basename(im_mask)

                mask = cv2.imread(im_mask, 0)
                mask = ((mask > 0) * 255).astype('uint8')

                assert len(np.unique(mask)) < 3, print(np.unique(mask))

                ret, labels = cv2.connectedComponents(mask)

                if ret > 1:
                    for l in range(ret):
                        if l == 0:
                            continue

                        ys, xs = np.where(labels == l)
                        x, y = xs.min(), ys.min()
                        w = xs.max() - xs.min()
                        h = ys.max() - ys.min()
                        gt.append([im_name, mask_name, classif, x, y, w, h])
                else:
                    gt.append([im_name, "", "NA", -1, -1, -1, -1])

    df = pd.DataFrame(gt, columns=["image", "mask", "class", "x", "y", "w", "h"])

    df.sort_values(by=['image'], inplace=True)
    df.to_csv(os.path.join("~/gt_{}.csv".format(dataset_name)), index=False)


# NAD = 1, AD = 2
def histology_testset(x):
    test_histos = {
        "001": 2,
        "002": 1,
        "003": 2,
        "004": 2,
        "005": 2,
        "006": 2,
        "007": 2,
        "008": 2,
        "009": 2,
        "010": 2,
        "011": 2,
        "012": 2,
        "013": 2,
        "014": 2,
        "015": 2,
        "016": 1,
        "017": 2,
        "018": 2,
    }
    seq = os.path.basename(x).split("-")[0]
    return test_histos[seq]


def histology_trainval(x):
    trainval_histos = {
        "001": 1,
        "002": 1,
        "003": 2,
        "004": 2,
        "005": 2,
        "006": 2,
        "007": 1,
        "008": 1,
        "009": 2,
        "010": 2,
        "011": 2,
        "012": 1,
        "013": 2,
        "014": 2,
        "015": 2,
        "016": 2,
        "017": 1,
        "018": 2,
    }
    seq = os.path.basename(x).split("-")[0]
    return trainval_histos[seq]


def histology_etis(x):
    return 2


def histology_hd(x):
    val = hd[hd.image == os.path.basename(x)]['class'].values
    if val.size == 0:
        print(x)
        return 2
    elif val[0] == 'AD':
        return 2
    else:
        return 0


if __name__ == '__main__':
    hd = pd.read_csv("../datasets/CVC_ClinicDB/gt.csv")
    calc_gt("Clinic", "../datasets/CVC_ClinicDB/images/", "bmp", "../datasets/CVC_ClinicDB/masks/", "tif", histology_hd)
    # hd = pd.read_csv("../datasets/CVC_ColonDB/gt.csv")
    # calc_gt("Colon", "../datasets/CVC_ColonDB/images/", "bmp", "../datasets/CVC_ColonDB/masks/", "bmp", histology_hd)
    # hd = pd.read_csv("../datasets/CVC_HDClassif/gt.csv")
    # calc_gt("HD", "../datasets/CVC_HDClassif/images/", "png", "../datasets/CVC_HDClassif/masks/", "tif", histology_hd)
    # calc_gt("train", "../datasets/CVC_VideoClinicDB_train/images/", "png", "../datasets/CVC_VideoClinicDB_train/masks/",
    #         "png", histology_trainval)
    # calc_gt("valid", "../datasets/CVC_VideoClinicDB_valid/images/", "png", "../datasets/CVC_VideoClinicDB_valid/masks/",
    #         "png", histology_trainval)
    # calc_gt("test", "../datasets/CVC_VideoClinicDB_test/images/", "png", "../datasets/CVC_VideoClinicDB_test/masks/",
    #         "png", histology_testset)
    # calc_gt("ETIS", "../datasets/ETIS-LaribPolypDB/images/", "tif", "../datasets/ETIS-LaribPolypDB/masks/", "tif",
    #         histology_etis)
