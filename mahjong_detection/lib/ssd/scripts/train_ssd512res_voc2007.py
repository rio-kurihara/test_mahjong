import os
import pickle
import numpy as np
from ssd.ssd import SingleShotMultiBoxDetector
from generator import Generator


if __name__ == "__main__":
    # settings
    labels = ["bg",  # first label should be bg.
              "aeroplane",
              "bus", "bicycle", "bird", "boat", "bottle",
              "car", "cat", "chair", "cow",
              "diningtable", "dog",
              "horse",
              "motorbike",
              "pottedplant", "person",
              "sheep", "sofa",
              "train", "tvmonitor"]
    n_classes = len(labels)
    input_shape = (512, 512, 3)

    aspect_ratios = [[2., 1/2.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2.],
                     [2., 1/2.]]
    scales = [(30., 60.),
              (60., 111.),
              (111., 162.),
              (162., 213.),
              (213., 264.),
              (264., 315.)]
    variances = [0.1, 0.1, 0.2, 0.2]

    # create network
    ssd = SingleShotMultiBoxDetector(model_type="ssd512",
                                     base_net="resnet50",
                                     n_classes=n_classes,
                                     class_names=labels,
                                     input_shape=input_shape,
                                     aspect_ratios=aspect_ratios,
                                     scales=scales,
                                     variances=variances,
                                     overlap_threshold=0.5,
                                     nms_threshold=0.45,
                                     max_output_size=400)
    # ssd.build(init_weight="./checkpoints/weights.29-1.16.hdf5")
    ssd.build()

    # make generator for training images
    batch_size = 32
    GROUND_TRUTH = "voc2007_annotations.pkl"
    INPUT_DATA = "../../../share/image-net/VOCdevkit/VOC2007/JPEGImages/"
    gt = pickle.load(open(GROUND_TRUTH, 'rb'))
    keys = sorted(gt.keys())
    indexes = np.arange(len(keys))
    np.random.seed(0)
    np.random.shuffle(indexes)
    num_train = int(round(0.8 * len(keys)))

    train_keys = np.array(keys)[indexes[:num_train]].tolist()
    val_keys = np.array(keys)[indexes[num_train:]].tolist()
    num_val = len(val_keys)
    gen = Generator(
        gt, ssd.bboxes, batch_size, INPUT_DATA,
        train_keys, val_keys,
        (input_shape[0], input_shape[1]),
        saturation_var=0.5, brightness_var=0.5,
        contrast_var=0.5, lighting_std=0.5,
        hflip_prob=0.5, vflip_prob=0.5,
        do_crop=False
    )

    # training
    path_to_checkpoints = os.sep.join((
        "./checkpoints/ssd512res_voc2007",
        "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    ))
    freeze = [
        'input_1',
        'conv1', 'bn_conv1',
        'res2a_branch2a', 'bn2a_branch2a', 'res2a_branch2b', 'bn2a_branch2b',
        'res2a_branch2c', 'res2a_branch1', 'bn2a_branch2c', 'bn2a_branch1',
        'res2b_branch2a', 'bn2b_branch2a', 'res2b_branch2b', 'bn2b_branch2b',
        'res2b_branch2c', 'bn2b_branch2c',
        'res2c_branch2a', 'bn2c_branch2a', 'res2c_branch2b', 'bn2c_branch2b',
        'res2c_branch2c', 'bn2c_branch2c',
        'res3a_branch2a', 'bn3a_branch2a', 'res3a_branch2b', 'bn3a_branch2b',
        'res3a_branch2c', 'res3a_branch1', 'bn3a_branch2c', 'bn3a_branch1',
        'res3b_branch2a', 'bn3b_branch2a', 'res3b_branch2b', 'bn3b_branch2b',
        'res3b_branch2c', 'bn3b_branch2c',
        'res3c_branch2a', 'bn3c_branch2a', 'res3c_branch2b', 'bn3c_branch2b',
        'res3c_branch2c', 'bn3c_branch2c',
        'res3d_branch2a', 'bn3d_branch2a', 'res3d_branch2b', 'bn3d_branch2b',
        'res3d_branch2c', 'bn3d_branch2c',
        'res4a_branch2a', 'bn4a_branch2a', 'res4a_branch2b', 'bn4a_branch2b',
        'res4a_branch2c', 'res4a_branch1', 'bn4a_branch2c', 'bn4a_branch1',
        'res4b_branch2a', 'bn4b_branch2a', 'res4b_branch2b', 'bn4b_branch2b',
        'res4b_branch2c', 'bn4b_branch2c',
        'res4c_branch2a', 'bn4c_branch2a', 'res4c_branch2b', 'bn4c_branch2b',
        'res4c_branch2c', 'bn4c_branch2c',
        'res4d_branch2a', 'bn4d_branch2a', 'res4d_branch2b', 'bn4d_branch2b',
        'res4d_branch2c', 'bn4d_branch2c',
        'res4e_branch2a', 'bn4e_branch2a', 'res4e_branch2b', 'bn4e_branch2b',
        'res4e_branch2c', 'bn4e_branch2c',
        'res4f_branch2a', 'bn4f_branch2a', 'res4f_branch2b', 'bn4f_branch2b',
        'res4f_branch2c', 'bn4f_branch2c',
        'res5a_branch2a', 'bn5a_branch2a', 'res5a_branch2b', 'bn5a_branch2b',
        'res5a_branch2c', 'res5a_branch1', 'bn5a_branch2c', 'bn5a_branch1',
        'res5b_branch2a', 'bn5b_branch2a', 'res5b_branch2b', 'bn5b_branch2b',
        'res5b_branch2c', 'bn5b_branch2c',
        'res5c_branch2a', 'bn5c_branch2a', 'res5c_branch2b', 'bn5c_branch2b',
        'res5c_branch2c', 'bn5c_branch2c',
              ]
    ssd.save_parameters("./checkpoints/ssd512res_voc2007/ssd512res_voc2007_params.json")
    ssd.train_by_generator(gen,
                           epoch=30,
                           learning_rate=1e-3,
                           neg_pos_ratio=3.0,
                           freeze=freeze,
                           checkpoints=path_to_checkpoints)
