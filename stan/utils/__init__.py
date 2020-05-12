# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


def freeze_model(model):
    for layer in model.layers:
        layer.trainable = True
