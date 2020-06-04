# Copyright (c) 2020 Hai Nguyen
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from .blocks import encoder_block, decoder_block, conv2D


def build_stan(
    n_classes,
    input_shape=(None, None, 3),
    filters=[32, 64, 128, 256, 512],
    decode_mode='transpose',
    activation='sigmoid'
):
    """Builder for pure STAN

    Parameters
    ----------
    n_classes : int
        Number of classes
    input_shape : tuple, optional
        Shape of input, by default (None, None, 3)
    filters : list, optional
        Declare number of filters for each encoding block,
        by default [32, 64, 128, 256, 512]
    decode_mode : str, optional
        [description], by default 'transpose'
    activation : str, optional
        activation function of the last layer before giving prediction,
        by default 'sigmoid'

    Returns
    -------
    Model
    """    

    inp = Input(shape=input_shape)

    # Encoder
    x3_pool, concat_pool, skip1_b1, skip2_b1 = encoder_block(
        filters[0])((inp, inp))

    x3_pool, concat_pool, skip1_b2, skip2_b2 = encoder_block(
        filters[1])((x3_pool, concat_pool))

    x3_pool, concat_pool, skip1_b3, skip2_b3 = encoder_block(
        filters[2])((x3_pool, concat_pool))

    x3_pool, concat_pool, skip1_b4, skip2_b4 = encoder_block(
        filters[3])((x3_pool, concat_pool))

    # middle
    x3_pool = conv2D(n_filters=filters[4], kernel_size=3)(x3_pool)
    x3_pool = conv2D(n_filters=filters[4], kernel_size=3)(x3_pool)
    concat_pool_1 = conv2D(n_filters=filters[4],
                           kernel_size=1)(concat_pool)
    concat_pool_1 = conv2D(n_filters=filters[4],
                           kernel_size=3)(concat_pool_1)
    concat_pool_5 = conv2D(n_filters=filters[4],
                           kernel_size=5)(concat_pool)
    concat_pool_5 = conv2D(n_filters=filters[4],
                           kernel_size=3)(concat_pool_5)
    mid = tf.concat([x3_pool, concat_pool_1, concat_pool_5],
                    axis=3, name='encoded_fm_concat')

    # Decoder
    x = decoder_block(n_filters=filters[3], 
                      mode=decode_mode)((mid, skip1_b4, skip2_b4))
    x = decoder_block(n_filters=filters[2],
                      mode=decode_mode)((x, skip1_b3, skip2_b3))
    x = decoder_block(n_filters=filters[1],
                      mode=decode_mode)((x, skip1_b2, skip2_b2))
    x = decoder_block(n_filters=filters[0],
                      mode=decode_mode)((x, skip1_b1, skip2_b1))

    # Last conv layer
    x = conv2D(n_filters=n_classes,
               kernel_size=3,
               activation=activation)(x)

    return Model(inp, x)
