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

        color = (255,0, 0) if loc_response in ["TP", "FP"] else None
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

            color = (0, 255,0) if loc_response in ["TP", "FP"] else None
            if color is not None:
                box = row.pred_box
                box = [int(float(x)) for x in box.strip("[]").split(",")]

                xy = (box[0], box[1])
                xy2 = (box[2], box[3])

                im = cv2.rectangle(im, xy, xy2, color, thickness=1)

        vid.write(im)

    vid.release()


def process_videos():
    last_seq = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('seq_{}.avi'.format(last_seq), fourcc, 30.0, (384, 288))
    for i, row in df.iterrows():
        if row.sequence != last_seq:
            vid.release()
            last_seq = row.sequence
            vid = cv2.VideoWriter('seq_{}.avi'.format(last_seq), fourcc, 30.0, (384, 288))

        im = cv2.imread(images_path + row.image)

        loc_response = row.localized

        color = get_color_by_score(row.score) if loc_response in ["TP", "FP"] else None
        if color is not None:
            box = row.pred_box
            box = [int(float(x)) for x in box.strip("[]").split(",")]

            xy = (box[0], box[1])
            xy2 = (box[2], box[3])

            im = cv2.rectangle(im, xy, xy2, color, thickness=2)

        vid.write(im)
    vid.release()


if __name__ == '__main__':
    model = "baselines/faster/faster_all_hd"
    input_file = "../results/{}/inference/giana/results.csv".format(model)
    images_path = "../datasets/CVC_VideoClinicDB_test/images/"
    df = pd.read_csv(input_file)

    model2 = "tests"
    input_b = input_file.replace(model, model2)
    # blue model  ; green model b
    compare_models(input_file, input_b, images_path)

    # process_videos()
