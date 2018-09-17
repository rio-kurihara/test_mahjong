import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

from keras.preprocessing import image
from PIL import Image
from scipy.misc import imread, imresize

from mahjong_detection.lib.ssd.ssd.ssd import SingleShotMultiBoxDetector
from mahjong_detection.lib.win_judgementer import WinJudgementer
from mahjong_detection.lib.point_calculater import PointCalculater
from mahjong_detection.lib.score_calculater import score_calculate

plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['image.interpolation'] = 'nearest'


THREDHOLD = 0.8

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

def _get_fname(save_dir):
    # 現在時刻からファイル名を作成
    save_fname = str(datetime.datetime.now())[:16].replace(' ', '_') + '.jpg'
    save_path = os.path.join(save_dir, save_fname)
    return save_path

def _load_file_from_s3():
    cmd = 'cd mahjong_detection/checkpoint\nwget https://s3-ap-northeast-1.amazonaws.com/test-mahjong/weights.25-0.05.hdf5'
    os.system(cmd)

def build_model():
    # build model
    model_file = 'mahjong_detection/checkpoint/weights.25-0.05.hdf5'
    param_file = 'mahjong_detection/checkpoint/ssd300_params_mahjong_vgg16_train_2.json'
    if not os.path.exists(model_file):
        _load_file_from_s3()

    ssd = SingleShotMultiBoxDetector(overlap_threshold=0.5, nms_threshold=0.45, max_output_size=400)
    ssd.load_parameters(param_file)
    ssd.build(init_weight=model_file)
    print('*'*40, 'model build')
    return ssd

def main(img, save_dir, ssd):
    input_shape = (512, 512, 3)

    inputs = []
    images = []

    img = image.img_to_array(img)
    img = add_margin(img)
    img = imresize(img, input_shape).astype("float32")
    images.append(img.copy())
    inputs.append(img.copy())

    inputs = np.array(inputs)

    results = ssd.detect(inputs, batch_size=1, verbose=1, do_preprocess=True)

    list_result_label = []
    list_result_score = []
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= THREDHOLD]

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
            list_result_label.append(label_name)
            list_result_score.append(round(score,2))

        save_path = _get_fname(save_dir)
        plt.savefig(save_path)
        plt.close()

        # 和了判定
        # wj = WinJudgementer(list_label)
        # eye, yaku = wj.agari()

        # 和了判定＋点計算
        # pc = PointCalculater(list_label, wj, index_seat_wind=33, index_round_wind=30, dora=3)
        # txt_dora, txt_han, wj.return_txt = pc.calc()
        # return txt_dora, txt_han, wj.return_txt
        # list_result_label = ['1m', '2m', '3m', '4p', '5p', '6p', '5s', '6s', '7s', 's', 's', 's', 'c', 'c']
        list_result_label = ['1m', '2m', '3m', 'f', 'f', '3s', '4s', '5s', 'n', 'n', 'n', '2p', '3p', '4p']
        print(list_result_label)
        return save_path, list_result_label

    # 点数計算
#     mark = 50
#     score_calculate(mark, point, flg_leader=False, flg_ron=True)
