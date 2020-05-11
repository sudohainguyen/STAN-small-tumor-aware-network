# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .builder import build_stan
from ..utils import freeze_model


def STAN(
    n_classes, 
    input_shape=(256, 256, 3),
    encoder_weights=None,
    output_activation='sigmoid',
    decode_mode='transpose',
    freeze_encoder=False,
    model_name='stan'
):
    """[summary]

    Arguments:
        n_classes {[type]} -- Number of classes for segmentation

    Keyword Arguments:
        input_shape {tuple} -- Shape of input tesors (default: {(256, 256, 3)})
        encoder_weights {[type]} -- pretrained weights path for encoder (default: {None})
        output_activation {str} -- Activation function for output prediction (default: {'sigmoid'})
        decode_mode {str} -- Mode for decoder, could be Transpose or Upsampling (default: {'transpose'})
        freeze_encoder {bool} -- Freezing encoder for fine-tuning (default: {False})
    """    
    model = build_stan(
        n_classes=n_classes,
        input_shape=input_shape,
        activation=output_activation,
        decode_mode=decode_mode)
    
    if freeze_encoder:
        freeze_model(model)
    
    # TODO: able to load pre-trained weights to encoder
    
    # model.name = model_name

    return model
