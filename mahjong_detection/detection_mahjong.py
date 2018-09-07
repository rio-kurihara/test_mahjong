import os
import datetime
import matplotlib
matplotlib.use('Agg')
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import sys
# sys.path.append('./')
from mahjong_detection.lib.ssd.ssd.ssd import SingleShotMultiBoxDetector

from mahjong_detection.lib.win_judgementer import WinJudgementer
from mahjong_detection.lib.point_calculater import PointCalculater
from mahjong_detection.lib.score_calculater import score_calculate
from collections import OrderedDict

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'


# def make_dir():
#     today = datetime.date.today()
#     save_dir = os.path.join('/home/rio.kurihara/mahjong/0616_test/mahjong_detector/result', str(today))
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     return save_dir


def add_margin(img):
    img_shape = list(img.shape)
    if img_shape[0] == img_shape[1]:
        return img
    if img_shape[0] < img_shape[1]:
        min_arg = 0
        max_arg = 1
    else:
        min_arg = 1
        max_arg = 0
    margin_shape = img_shape
    margin_shape[min_arg] = int((img_shape[max_arg] - img_shape[min_arg])/2.)
    margin = np.tile([0.], margin_shape)
    new_img = np.concatenate([margin, img], axis=min_arg)
    new_img = np.concatenate([new_img, margin], axis=min_arg)
    return new_img

def main(img_path):
    # load model
    model_file = './checkpoint/weights.25-0.05.hdf5'
    param_file = './checkpoint/ssd300_params_mahjong_vgg16_train_2.json'
    ssd = SingleShotMultiBoxDetector(overlap_threshold=0.5, nms_threshold=0.45, max_output_size=400)
    ssd.load_parameters(param_file)
    ssd.build(init_weight=model_file)

    input_shape = (512, 512, 3)

    inputs = []
    images = []

    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = add_margin(img)
    img = imresize(img, input_shape).astype("float32")
    images.append(img.copy())
    inputs.append(img.copy())

    inputs = np.array(inputs)

    results = ssd.detect(inputs, batch_size=1, verbose=1, do_preprocess=True)

    list_label = []
    import pandas as pd
    for i, img in enumerate(images):
        result = pd.DataFrame()

        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.9]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 35)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = ssd.class_names[label]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':1.0})
            # 結果格納
            list_result = [label_name, round(score,2)]
            tmp_result = pd.DataFrame(list_result).T
            result = pd.concat([result, tmp_result], axis=0)

            list_label.append(label_name)
        # save
    #     save_dir = make_dir()
    #     jstTime = datetime.datetime.now() + datetime.timedelta(hours=9)
    #     save_fname = str(jstTime.time())[:5] + '.png'
    #     plt.savefig(os.path.join(save_dir, save_fname))
    # #     plt.show()
        result.columns = ['pi_name', 'max_match_val']

        # 和了判定
        wj = WinJudgementer(list_label)
        # eye, yaku = wj.agari()

        # 和了判定＋点計算
        pc = PointCalculater(list_label, wj, index_seat_wind=33, index_round_wind=30, dora=3)
        txt_dora, txt_han, wj.return_txt = pc.calc()
        return txt_dora, txt_han, wj.return_txt

    # 点数計算
#     mark = 50
#     score_calculate(mark, point, flg_leader=False, flg_ron=True)
