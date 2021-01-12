import os
import cv2
import pandas as pd
from tqdm import tqdm


def get_color_by_score(score):
    if score < 0.5:
        return None
    if score < 0.6:
        return (0, 0, 255)
    if score < 0.8:
        return (0, 215, 255)
    else:
        return (0, 255, 0)


# only if model has 1 det per frame
# TODO generalize to all models (iter by image and filter both)
def compare_models(in_a, in_b, dataset_images_path):
    df_a = pd.read_csv(in_a)
    df_b = pd.read_csv(in_b)

    last_seq = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('seq_{}.avi'.format(last_seq), fourcc, 30.0, (384, 288))

    for i, row in tqdm(df_a.iterrows()):
        if row.sequence != last_seq:
            vid.release()
            last_seq = row.sequence
            vid = cv2.VideoWriter('seq_{}.avi'.format(last_seq), fourcc, 30.0, (384, 288))

        im = cv2.imread(dataset_images_path + row.image)

        loc_response = row.localized

        # blue
        color = (255, 0, 0) if loc_response in ["TP", "FP"] else None
        if color is not None:
            box = row.pred_box
            box = [int(float(x)) for x in box.strip("[]").split(",")]

            xy = (box[0], box[1])
            xy2 = (box[2], box[3])

            im = cv2.rectangle(im, xy, xy2, color, thickness=2)

        fi_b = df_b[df_b.image == row.image]

        for i, row in fi_b.iterrows():
            if row.sequence != last_seq:
                vid.release()
                last_seq = row.sequence
                vid = cv2.VideoWriter('seq_{}.avi'.format(last_seq), fourcc, 30.0, (384, 288))

            loc_response = row.localized

            # green
            color = (0, 255,0) if loc_response in ["TP", "FP"] else None
            if color is not None:
                box = row.pred_box
                box = [int(float(x)) for x in box.strip("[]").split(",")]

                xy = (box[0], box[1])
                xy2 = (box[2], box[3])

                im = cv2.rectangle(im, xy, xy2, color, thickness=1, lineType=4)

        vid.write(im)

    vid.release()


def process_videos(df, out="vids"):
    os.makedirs(out, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = None
    sequences = df.sequence.unique()
    for seq in sequences:
        if vid is None:
            vid = cv2.VideoWriter('{}/seq_{}.avi'.format(out, seq), fourcc, 30.0, (384, 288))

        filtered_df = df[df['sequence'] == seq]
        for frame in df[df['sequence'] == seq].frame.unique():
            rows = filtered_df[filtered_df.frame == frame]
            im = cv2.imread(images_path + rows.image.unique()[0])

            for i, row in rows.iterrows():
                loc_response = row.localized

                # color = get_color_by_score(row.score) if loc_response in ["TP", "FP"] else None
                color = (0, 0, 255) if loc_response in ["TP", "FP"] and row.score > 0.5 else None
                if color is not None:
                    box = row.pred_box
                    box = [int(float(x)) for x in box.strip("[]").split(",")]

                    xy = (box[0], box[1])
                    xy2 = (box[2], box[3])

                    im = cv2.rectangle(im, xy, xy2, color, thickness=2)

            vid.write(im)

        vid.release()
        vid = None


    # for i, row in df.iterrows():
    #     if row.sequence != last_seq:
    #         vid.release()
    #         last_seq = row.sequence
    #         vid = cv2.VideoWriter('{}/seq_{}.avi'.format(out, last_seq), fourcc, 30.0, (384, 288))
    #
    #     im = cv2.imread(images_path + row.image)
    #
    #     loc_response = row.localized
    #
    #     color = get_color_by_score(row.score) if loc_response in ["TP", "FP"] else None
    #     if color is not None:
    #         box = row.pred_boxre
    #         box = [int(float(x)) for x in box.strip("[]").split(",")]
    #
    #         xy = (box[0], box[1])
    #         xy2 = (box[2], box[3])
    #
    #         im = cv2.rectangle(im, xy, xy2, color, thickness=2)
    #
    #     vid.write(im)
    # vid.release()


def from_ims_to_vid(images_path, output_path, n_seq=18):
    os.makedirs(output_path, exist_ok=True)
    from glob import glob

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for seq in range(1, n_seq + 1):
        ims = sorted(glob(images_path + "/{:03d}-*.png".format(seq)))
        vid = cv2.VideoWriter('{}/seq_{}.avi'.format(output_path, seq), fourcc, 25.0, (768, 288))

        for i in ims:
            vid.write(cv2.imread(i))
        vid.release()


if __name__ == '__main__':
    # input_dir = "results/results/all_exps/final/m101_concat_sam_2x/inference/images/"
    # output_dir = "results/results/all_exps/final/m101_concat_sam_2x/inference/vids/"

    # from_ims_to_vid(input_dir, output_dir)
    #
    # dsfsadf
    #
    input_file = "/media/devsodin/WORK1/TFM/results/all_exps/final/f101_losses_concat_double_2x/inference/giana_CVC_VideoClinicDB_test/results.csv"
    images_path = "/media/devsodin/WORK1/datasets/CVC_VideoClinicDB_test/images/"
    df = pd.read_csv(input_file)
    #
    # model2 = "refine_cls/faster_post"
    # input_b = input_file.replace(model, model2)
    # blue model  a; green model b
    # compare_models(input_file, input_b, images_path)
    #
    process_videos(df)
