# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Activation, BatchNormalization, 
    MaxPooling2D, Layer, 
    UpSampling2D, Conv2DTranspose
)


class Conv2DBlock(Layer):
    def __init__(
        self,
        n_filters,
        kernel_size,
        activation='relu',
        use_bn=False,
        name='conv2d',
        **kwargs
    ):
        super(Conv2DBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn
        self.conv = Conv2D(n_filters, kernel_size, name=f'{name}_conv',
                           use_bias=(not use_bn), padding='same', **kwargs)
        self.use_bn = use_bn
        if use_bn:
            # self.bn = BatchNorm(name=f'{name}_bn')
            self.bn = BatchNormalization(name=f'{name}_bn')
        self.activation = Activation(activation, name=f'{name}_activation')
        
    def call(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(Conv2DBlock, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'use_bn': self.use_bn
        })
        return config


class BatchNorm(BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class EncoderBlock(Layer):
    def __init__(self, n_filters, name='enc'):
        super(EncoderBlock, self).__init__()
        self.n_filters = n_filters
        self.conv3_1 = Conv2DBlock(n_filters=n_filters,
                                   kernel_size=3,
                                   name=f'{name}_kernel3_1')
        self.conv3_2 = Conv2DBlock(n_filters=n_filters,
                                   kernel_size=3,
                                   name=f'{name}_kernel3_2')
        self.conv3_pool = MaxPooling2D(pool_size=(2, 2),
                                       name=f'{name}_kernel3_pool')

        self.conv1_1 = Conv2DBlock(n_filters=n_filters,
                                   kernel_size=1,
                                   name=f'{name}_kernel1_1')
        self.conv1_2 = Conv2DBlock(n_filters=n_filters,
                                   kernel_size=1,
                                   name=f'{name}_kernel1_2')

        self.conv5_1 = Conv2DBlock(n_filters=n_filters,
                                   kernel_size=5,
                                   name=f'{name}_kernel5_1')
        self.conv5_2 = Conv2DBlock(n_filters=n_filters,
                                   kernel_size=5,
                                   name=f'{name}_kernel5_1')
        self.agg_pool = MaxPooling2D(pool_size=(2, 2),
                                     name=f'{name}_agg_pool')

    def call(self, inputs):
        """Feed forward through the Encoder block

        Arguments:
            kernel3_inp {tensor} -- feature maps of kernel3 conv
                from previous block or just an image
            kernelconcat_inp {[type]} -- feature maps of concatenation
                (kernel1 + kernel5)
                from previous block or just an image

        Returns:
            tuple -- (kernel3 output, concat output,
                      skip connection 1, skip connection 2)
        """
        kernel3_inp, kernelconcat_inp = inputs
        
        x1 = self.conv1_1(kernelconcat_inp)
        x1 = self.conv1_2(x1)

        x5 = self.conv5_1(kernelconcat_inp)
        x5 = self.conv5_2(x5)

        concat = tf.concat([x1, x5], axis=3)
        concat_pool = self.agg_pool(concat)

        x3 = x3_1 = self.conv3_1(kernel3_inp)
        x3 = skip1 = self.conv3_2(x3)
        x3_pool = self.conv3_pool(x3)

        skip2 = tf.concat([x3_1, concat], axis=3)
        return x3_pool, concat_pool, skip1, skip2

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """        
        config = super(EncoderBlock, self).get_config()
        config.update({
            'n_filters': self.n_filters
        })
        return config


class DecoderBlock(Layer):
    def __init__(
        self, 
        n_filters, 
        mode='upsampling'
    ):
        """Initialize Decoding Block using UpSample layers

        Arguments:
            n_filters {int} -- number of filters for each conv2d layers

        Keyword Arguments:
            mode {str} -- Decoding mode (default: {'upsampling'})

        Raises:
            ValueError: when the mode is not 'upsampling' or 'transpose'
        """        

        super(DecoderBlock, self).__init__()
        self.n_filters = n_filters
        self.mode = mode
        self.conv_1 = Conv2DBlock(n_filters, kernel_size=3)
        self.conv_2 = Conv2DBlock(n_filters, kernel_size=3)

        if mode == 'upsampling':
            self.up = UpSampling2D(size=(2, 2))
        elif mode == 'transpose':
            self.up = Conv2DTranspose(n_filters, kernel_size=3,
                                      strides=(2, 2), padding='same')
        else:
            raise ValueError()

    def call(self, inputs):
        """Feed forward the Decoding block

        Arguments:
            inp {tensor} -- the main input
            skip1 {tensor} -- feature maps from skip connection 1
            skip2 {tensor} -- feature maps from skip connection 2

        Returns:
            tensor -- output
        """
        inp, skip1, skip2 = inputs
        
        x = self.up(inp)
        x = self.conv_1(tf.concat([x, skip1], axis=3))
        x = self.conv_2(tf.concat([x, skip2], axis=3))
        return x

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """        
        config = super(DecoderBlock, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'mode': self.mode
        })
        return config
