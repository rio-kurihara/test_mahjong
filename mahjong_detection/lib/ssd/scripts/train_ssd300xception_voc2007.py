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
    input_shape = (300, 300, 3)

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
    ssd = SingleShotMultiBoxDetector(model_type="ssd300",
                                     base_net="xception",
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
        "./checkpoints",
        "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    ))
    freeze = [
        'input_1',
        'block1_conv1', 'block1_conv1_bn', 'block1_conv2', 'block1_conv2_bn',
        'block2_sepconv1', 'block2_sepconv1_bn', 'block2_sepconv2', 'block2_sepconv2_bn',
        'conv2d_1', 'block2_pool', 'batch_normalization_1',
        'block3_sepconv1', 'block3_sepconv1_bn', 'block3_sepconv2', 'block3_sepconv2_bn',
        'conv2d_2', 'block3_pool', 'batch_normalization_2',
        'block4_sepconv1', 'block4_sepconv1_bn', 'block4_sepconv2', 'block4_sepconv2_bn',
        'conv2d_3', 'block4_pool', 'batch_normalization_3',
        'block5_sepconv1', 'block5_sepconv1_bn', 'block5_sepconv2', 'block5_sepconv2_bn',
        'block5_sepconv3', 'block5_sepconv3_bn',
        'block6_sepconv1', 'block6_sepconv1_bn', 'block6_sepconv2', 'block6_sepconv2_bn',
        'block6_sepconv3', 'block6_sepconv3_bn',
        'block7_sepconv1', 'block7_sepconv1_bn', 'block7_sepconv2', 'block7_sepconv2_bn',
        'block7_sepconv3', 'block7_sepconv3_bn',
        'block8_sepconv1', 'block8_sepconv1_bn', 'block8_sepconv2', 'block8_sepconv2_bn',
        'block8_sepconv3', 'block8_sepconv3_bn',
        'block9_sepconv1', 'block9_sepconv1_bn', 'block9_sepconv2', 'block9_sepconv2_bn',
        'block9_sepconv3', 'block9_sepconv3_bn',
        'block10_sepconv1', 'block10_sepconv1_bn', 'block10_sepconv2', 'block10_sepconv2_bn',
        'block10_sepconv3', 'block10_sepconv3_bn',
        'block11_sepconv1', 'block11_sepconv1_bn', 'block11_sepconv2', 'block11_sepconv2_bn',
        'block11_sepconv3', 'block11_sepconv3_bn',
        'block12_sepconv1', 'block12_sepconv1_bn', 'block12_sepconv2', 'block12_sepconv2_bn',
        'block12_sepconv3', 'block12_sepconv3_bn',
        'block13_sepconv1', 'block13_sepconv1_bn', 'block13_sepconv2', 'block13_sepconv2_bn',
        'conv2d_4', 'block13_pool', 'batch_normalization_4',
        'block14_sepconv1', 'block14_sepconv1_bn', 'block14_sepconv2', 'block14_sepconv2_bn',
        # 'block9_conv1', 'block9_conv2',
        # 'block10_conv1', 'block10_conv2',
        # 'block11_conv1', 'block11_conv2',
        # 'block4_sepconv2_norm',
    ]
    ssd.save_parameters("./checkpoints/ssd300_voc2007_params.json")
    ssd.train_by_generator(gen,
                           epoch=30,
                           learning_rate=1e-3,
                           neg_pos_ratio=3.0,
                           freeze=freeze,
                           checkpoints=path_to_checkpoints)
