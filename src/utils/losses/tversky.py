# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow.keras.backend as K


def _tversky_index(y_true, y_pred):
    smooth = 1.0
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    alpha = 0.7

    return (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - _tversky_index(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = _tversky_index(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)
