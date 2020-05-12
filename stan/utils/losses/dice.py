# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from ..metrics import dice_coef


def dice_loss(smooth_dice=1.0):
    def compute_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred, smooth=smooth_dice)
    return compute_loss
