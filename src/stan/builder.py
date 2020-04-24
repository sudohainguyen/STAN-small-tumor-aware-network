# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer, UpSampling2D, Input

from .blocks import EncoderBlock, DecoderBlock, Conv2DBlock


def build_stan(
    n_classes,
    input_shape=(256, 256, 3),
    filters=[32, 64, 128, 256, 512],
    activation='sigmoid'
):
    """Pure STAN builder using Conv blocks (no backbone)

    Arguments:
        n_classes {int} -- Number of classes

    Keyword Arguments:
        input_shape {tuple} -- The shape of input (default: {(256, 256, 3)})
        filters {list} -- Declare number of filters for each encoding block (default: {[32, 64, 128, 256, 512]})
        activation {str} -- activation function of the last layer before giving prediction
    """
    # inp = InputLayer(input_shape=input_shape)
    inp = Input(shape=input_shape)
    
    # Encoder
    x3_pool, concat_pool, skip1_b1, skip2_b1 = EncoderBlock(filters[0])(
        kernel3_inp=inp, kernelconcat_inp=inp)
    
    x3_pool, concat_pool, skip1_b2, skip2_b2 = EncoderBlock(filters[1])(
        kernel3_inp=x3_pool, kernelconcat_inp=concat_pool)
    
    x3_pool, concat_pool, skip1_b3, skip2_b3 = EncoderBlock(filters[2])(
        kernel3_inp=x3_pool, kernelconcat_inp=concat_pool)
    
    x3_pool, concat_pool, skip1_b4, skip2_b4 = EncoderBlock(filters[3])(
        kernel3_inp=x3_pool, kernelconcat_inp=concat_pool)
    
    # middle
    x3_pool         = Conv2DBlock(n_filters=filters[4], kernel_size=3)(x3_pool)
    x3_pool         = Conv2DBlock(n_filters=filters[4], kernel_size=3)(x3_pool)
    concat_pool_1   = Conv2DBlock(n_filters=filters[4], kernel_size=1)(concat_pool)
    concat_pool_1   = Conv2DBlock(n_filters=filters[4], kernel_size=3)(concat_pool)
    concat_pool_5   = Conv2DBlock(n_filters=filters[4], kernel_size=5)(concat_pool)
    concat_pool_5   = Conv2DBlock(n_filters=filters[4], kernel_size=3)(concat_pool)

    mid = tf.concat([x3_pool, concat_pool_1, concat_pool_5], axis=3)
    
    # Decoder
    x = DecoderBlock(n_filters=filters[3])(mid, skip1_b4, skip2_b4)
    x = DecoderBlock(n_filters=filters[2])(x, skip1_b3, skip2_b3)
    x = DecoderBlock(n_filters=filters[1])(x, skip1_b2, skip2_b2)
    x = DecoderBlock(n_filters=filters[0], is_last_block=True)(x, skip1_b1, skip2_b1)
    
    # Last conv layer
    x = Conv2DBlock(n_filters=n_classes, kernel_size=3, activation=activation)(x)

    return Model(inp, x)

# def build_stan_with_backbone(
#     n_classes,
#     backbone='vgg'
# ):
#     pass
