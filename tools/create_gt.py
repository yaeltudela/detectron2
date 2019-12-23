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


def calc_gt(images_folder, ims_ext, masks_folder, masks_ext, check_histology):
    df = pd.DataFrame(columns=["image", "mask", "has_polyp", "class", "x_min", "y_min", "x_max", "y_max", "center_x", "center_y"])

    def add_row(row):
        df.loc[len(df)] = row
        df.index += 1
        df.reset_index(inplace=True, drop=True)

    ims = glob(os.path.join(images_folder, f"*.{ims_ext}"))
    masks = glob(os.path.join(masks_folder, f"*.{masks_ext}"))

    for image in ims:
        im_name = os.path.basename(image)

        im_masks = get_valid_masks(image, masks)
        if not im_masks:
            add_row([im_name, "", 0, "", -1, -1, -1, -1, -1, -1])
        else:
            for im_mask in im_masks:
                remove_original = False
                classif = check_histology(image)
                mask_name = os.path.basename(im_mask)

                mask = cv2.imread(im_mask, 0)
                ret, labels = cv2.connectedComponents(mask)

                if ret > 1:
                    for l in range(1, ret):
                        xs, ys = np.where(labels == l)
                        cx = xs.min() + (xs.max() - xs.min()) / 2
                        cy = ys.min() + (ys.max() - ys.min()) / 2

                        if ret > 2:
                            nname = mask_name.replace(".tif", "_{}.tif".format(l))
                            nmask = labels * (labels == l) * 255 / l
                            nmask = nmask
                            cv2.imwrite(nname, nmask.astype("uint8"))
                            remove_original = True

                        add_row([im_name, mask_name, 1, classif, xs.min(), ys.min(), xs.max(), ys.max(), int(cx), int(cy)])

                else:
                    add_row([im_name, "", 0, "", -1, -1, -1, -1, -1, -1])

                if remove_original:
                    print("removing {}".format(im_mask))
                    os.remove(im_mask)
    df.sort_values(by=['image'], inplace=True)
    df[['sequence', 'frame']] = df.image.str.split("-", expand=True)
    df.frame = df.frame.map(lambda x : x.split(".")[0])
    df.to_csv("gt.csv", index=False)


def check_histology(x):
    return x


def histology_testset(x):
    seq = os.path.basename(x).split("-")[0]
    return test_histos[seq]


if __name__ == '__main__':
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

    calc_gt("../datasets/cvcvideoclinicdbtest/images/", "png", "../datasets/cvcvideoclinicdbtest/masks/", "tif",
            histology_testset)
