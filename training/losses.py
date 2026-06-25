import tensorflow as tf

from config import (
    LOSS_DICE_WEIGHT,
    LOSS_FOCAL_WEIGHT,
    LOSS_CE_WEIGHT,
    FOCAL_GAMMA,
    NUM_CLASSES,
)


def dice_loss(y_true, y_pred, smooth=1e-6):
    axes = [1, 2]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)


def focal_loss(y_true, y_pred, gamma=FOCAL_GAMMA, eps=1e-7):
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = tf.pow(1.0 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def cross_entropy_loss(y_true, y_pred, eps=1e-7):
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    loss = -y_true * tf.math.log(y_pred)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def composite_loss(y_true, y_pred):
    d = dice_loss(y_true, y_pred)
    f = focal_loss(y_true, y_pred)
    c = cross_entropy_loss(y_true, y_pred)
    return LOSS_DICE_WEIGHT * d + LOSS_FOCAL_WEIGHT * f + LOSS_CE_WEIGHT * c
