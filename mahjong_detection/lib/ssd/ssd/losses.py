import tensorflow as tf
from keras.losses import categorical_crossentropy


class MultiBoxLoss:
    """
    """
    def __init__(self, n_classes, alpha=1.0, neg_pos_ratio=3.0,
                 negatives_for_hard=100):
        self.n_classes = n_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.negatives_for_hard = negatives_for_hard

    def _softmax_loss(self, y_true, y_pred):
        """
        """
        softmax_loss = categorical_crossentropy(y_true, y_pred)
        # y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        # softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return softmax_loss

    def _l1_smooth_loss(self, y_true, y_pred):
        """
        """
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def compute_loss_old(self, y_true, y_pred):
        """ compute loss
        """
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # loss for all default boxes
        conf_loss = self._softmax_loss(y_true[:, :, 4:],
                                       y_pred[:, :, 4:])
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # positives loss
        num_pos = num_boxes - tf.reduce_sum(y_true[:, :, 4], axis=-1)
        fpmask = 1 - y_true[:, :, 4]
        pos_loc_loss = tf.reduce_sum(loc_loss * fpmask, axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * fpmask, axis=1)

        # negatives loss
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0,
                            values=[num_neg,
                                    [(1 - has_min) * self.negatives_for_hard]])
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        confs_start = 4 + 1
        confs_end = confs_start + self.n_classes - 1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)

        nvalues, indices = tf.nn.top_k(max_confs * y_true[:, :, 4],
                                       k=num_neg_batch)

        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                        tf.reshape(indices, [-1]))

        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # loss is sum of positives and negatives
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                           tf.ones_like(num_pos))
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        return total_loss

    def compute_loss(self, y_true, y_pred):
        """ compute loss
        """
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # loss for all default boxes
        conf_loss = self._softmax_loss(y_true[:, :, 4:],
                                       y_pred[:, :, 4:])
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # positives loss
        num_pos = num_boxes - tf.reduce_sum(y_true[:, :, 4], axis=-1)
        fpmask = 1 - y_true[:, :, 4]
        pos_loc_loss = tf.reduce_sum(loc_loss * fpmask, axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * fpmask, axis=1)

        # negatives loss
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0,
                            values=[num_neg,
                                    [(1 - has_min) * self.negatives_for_hard]])
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        confs_start = 4 + 1
        confs_end = confs_start + self.n_classes - 1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)

        nvalues, indices = tf.nn.top_k(max_confs * y_true[:, :, 4],
                                       k=num_neg_batch)
        min_nvalues = nvalues[:, -1]
        min_nvalues = tf.expand_dims(min_nvalues, 1)
        min_nvalues = tf.tile(min_nvalues, (1, tf.shape(max_confs)[1]))
        nmask = tf.logical_not(tf.cast(fpmask, tf.bool))
        nmask = tf.logical_and(nmask,
                               tf.greater_equal(max_confs, min_nvalues))
        fnmask = tf.to_float(nmask)

        neg_conf_loss = tf.reduce_sum(conf_loss * fnmask, axis=1)

        # loss is sum of positives and negatives
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                           tf.ones_like(num_pos))
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        return total_loss
