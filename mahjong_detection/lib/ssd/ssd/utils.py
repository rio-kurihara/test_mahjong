import os
import tensorflow as tf
import numpy as np
from xml.etree import ElementTree


def make_bboxes(input_shape, feature_map_shape,
                aspect_ratios, scale):
    """
    """
    map_w = feature_map_shape[1]
    map_h = feature_map_shape[2]
    input_w = input_shape[0]
    input_h = input_shape[1]

    # local box's sizes
    min_size = scale[0]
    box_w = [min_size]
    box_h = [min_size]
    if len(scale) == 2:
        box_w.append(np.sqrt(min_size * scale[1]))
        box_h.append(np.sqrt(min_size * scale[1]))
    for ar in aspect_ratios:
        box_w.append(min_size * np.sqrt(ar))
        box_h.append(min_size / np.sqrt(ar))
    box_w = np.array(box_w)/2/input_w
    box_h = np.array(box_h)/2/input_h

    # feature grids
    step_w = input_w / map_w
    step_h = input_h / map_h
    center_h, center_w = np.mgrid[0:map_w, 0:map_h] + 0.5
    # swap h and w due to after reshapes
    center_w = (center_w * step_w/input_w).reshape(-1, 1)
    center_h = (center_h * step_h/input_h).reshape(-1, 1)

    n_local_box = len(box_w)
    bboxes = np.concatenate((center_w, center_h), axis=1)
    bboxes = np.tile(bboxes, (1, 2 * n_local_box))
    bboxes[:, ::4] -= box_w
    bboxes[:, 1::4] -= box_h
    bboxes[:, 2::4] += box_w
    bboxes[:, 3::4] += box_h
    bboxes = bboxes.reshape(-1, 4)
    bboxes = np.minimum(np.maximum(bboxes, 0.0), 1.0)

    return bboxes


class BoundaryBox:
    """
    """
    def __init__(self, n_classes, default_boxes, variances,
                 overlap_threshold=0.5, nms_threshold=0.45,
                 max_output_size=400):
        self.n_classes = n_classes
        self.default_boxes = default_boxes
        self.variances = np.array(variances)
        self.overlap_threshold = overlap_threshold
        self.nms_threshold = nms_threshold
        self.max_output_size = max_output_size

        self.n_boxes = 0 if default_boxes is None else len(default_boxes)
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(
            self.boxes, self.scores,
            self.max_output_size,
            iou_threshold=self.nms_threshold
        )
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    def iou(self, box):
        """Compute intersection over union for the box with all default_boxes.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_default_boxes).
        """
        # compute intersection
        inter_upleft = np.maximum(self.default_boxes[:, :2], box[:2])
        inter_botright = np.minimum(self.default_boxes[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.default_boxes[:, 2] - self.default_boxes[:, 0])
        area_gt *= (self.default_boxes[:, 3] - self.default_boxes[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode(self, box, return_iou=True):
        """Encode ground truth box into default boxes for training.
        Args:
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        Returns:
            encoded_box: Tensor with encoded box
                numpy tensor of shape (n_boxes, 4 + int(return_iou)).
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.n_boxes, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_default_boxes = self.default_boxes[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_default_boxes_center = 0.5 * (assigned_default_boxes[:, :2] +
                                               assigned_default_boxes[:, 2:4])
        assigned_default_boxes_wh = (assigned_default_boxes[:, 2:4] -
                                     assigned_default_boxes[:, :2])
        # we encode variance
        encoded_box[:, :2][assign_mask] = \
            box_center - assigned_default_boxes_center
        encoded_box[:, :2][assign_mask] /= assigned_default_boxes_wh
        encoded_box[:, :2][assign_mask] /= np.array([self.variances[:2]])
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_default_boxes_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array([self.variances[2:]])
        return encoded_box

    def assign_boxes(self, boxes):
        """ Assign boxes into default box for training.

        Args:
            boxes: Box, numpy tensor of shape (n_boxes, 4 + n_classes),
                n_classes is NOT including background.

        Returns:
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (n_boxes, 4 + n_classes),
                n_classes in inlucding background.
                assigment[4] indicates BG confidence.
        """
        assignment = np.zeros((self.n_boxes, 4 + self.n_classes))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode, 1, boxes[:, :4])
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:][best_iou_mask] = boxes[best_iou_idx, 4:]
        return assignment

    def decode(self, mbox_loc):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            mbox_loc: Numpy array of predicted locations.

        # Return
            decode_bbox: Shifted priors.
        """
        prior_width = self.default_boxes[:, 2] - self.default_boxes[:, 0]
        prior_height = self.default_boxes[:, 3] - self.default_boxes[:, 1]
        prior_center_x = \
            0.5 * (self.default_boxes[:, 2] + self.default_boxes[:, 0])
        prior_center_y = \
            0.5 * (self.default_boxes[:, 3] + self.default_boxes[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * self.variances[0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * self.variances[1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * self.variances[2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * self.variances[3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, keep_top_k=200,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        mbox_conf = predictions[:, :, 4:]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode(mbox_loc[i])
            for c in range(self.n_classes):
                if c == 0:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results


class VOCAnnotationReader(object):

    def __init__(self, data_path, label_names):
        self.path_prefix = data_path
        self.label_names = label_names
        self.num_classes = len(label_names)
        self.data = dict()
        self._read_xml()

    def _read_xml(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(os.sep.join((
                self.path_prefix, filename
            )))
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin, ymin, xmax, ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name in self.label_names:
            index = self.label_names.index(name)
            one_hot_vector[index] = 1
        else:
            print('unknown label: %s' % name)

        return one_hot_vector
