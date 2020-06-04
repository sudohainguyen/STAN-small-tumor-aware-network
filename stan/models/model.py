# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from stan.utils import freeze_model
from .builder import build_stan


# def freeze_model(model):
#     for layer in model.layers:
#         layer.trainable = True


def STAN(
    n_classes, 
    input_shape=(None, None, 3),
    filters=[32, 64, 128, 256, 512],
    output_activation='sigmoid',
    decode_mode='transpose',
    freeze_encoder=False,
    model_name='stan'
):
    """Initialize STAN model

    Parameters
    ----------
    n_classes : int
        Number of classes
    input_shape : tuple, optional
        Shape of input tesors, by default (None, None, 3)
    filters : list, optional
        Number of filters for each encoder block,
        by default [32, 64, 128, 256, 512]
    output_activation : str, optional
        Activation function for output prediction, by default 'sigmoid'
    decode_mode : str, optional
        Mode for decoder, could be `transpose` or `upsampling`,
        by default 'transpose'
    freeze_encoder : bool, optional
        Freezing encoder for fine-tuning, by default False
    model_name : str, optional
        Model name, by default 'stan'

    Returns
    -------
    Model
    """    
    model = build_stan(
        n_classes=n_classes,
        input_shape=input_shape,
        filters=filters,
        activation=output_activation,
        decode_mode=decode_mode)
    
    if freeze_encoder:
        freeze_model(model)
    
    # TODO: able to load pre-trained weights to encoder
    
    # model.name = model_name

    return model
