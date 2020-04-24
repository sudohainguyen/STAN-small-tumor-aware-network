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
    freeze_encoder=False,
    model_name='stan'
):
    """[summary]

    Arguments:
        n_classes {[type]} -- [description]

    Keyword Arguments:
        input_shape {tuple} -- [description] (default: {(256, 256, 3)})
        encoder_weights {[type]} -- [description] (default: {None})
        output_activation {str} -- [description] (default: {'sigmoid'})
        freeze_encoder {bool} -- [description] (default: {False})
    """    
    model = build_stan(n_classes=n_classes, input_shape=input_shape, activation=output_activation)
    
    if freeze_encoder:
        freeze_model(model)
    
    # TODO: able to load pre-trained weights to encoder
    
    model.name = model_name

    return model