import tensorflow as tf
import numpy as np
from training.resnet import resnet_v1
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')
FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    with tf.variable_scope("Unpooling_layer"):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images, means=None):
    """
    image normalization
    Args:
        images: tf tensor
        means: mean to subtract. Default [123.68, 116.78, 103.94]
    Returns: tf tensor
    """
    with tf.variable_scope("Mean_image_substraction"):
        if means is None:
            means = [123.68, 116.78, 103.94]

        num_channels = images.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    """
    define the model, we use slim's implementation of resnet
    """
    images = mean_image_subtraction(images)
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            # Notations f, g, h are explained in the paper, figure 3
            f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    # See paper
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            f_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope="score_map")
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = tf.multiply(slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None),
                                  FLAGS.text_scale, name="geo_map")
            angle_map = tf.multiply(slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid,
                                                normalizer_fn=None) - 0.5, np.pi,
                                    name="angle")  # angle is between [-90, 90]
            f_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return f_score, f_geometry


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss
    Args:
        y_true_cls: ground-truth
        y_pred_cls: predicted
        training_mask:
    Returns: The loss

    """
    with tf.variable_scope("Dice_coefficient"):
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        tf.summary.scalar('classification_dice_loss', loss)
        return loss


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    """
    define the loss used for training, containing two part:
      - the first part use dice loss instead of weighted logloss,
      - the second part is the iou loss defined in the paper
    Args:
        y_true_cls: ground truth of text
        y_pred_cls: prediction of text
        y_true_geo: ground truth of geometry
        y_pred_geo: prediction of geometry
        training_mask: mask used in training, to ignore some text annotated by ###
    Returns:

    """
    with tf.variable_scope("Loss"):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)

        # TODO why + ????
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta

        return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
