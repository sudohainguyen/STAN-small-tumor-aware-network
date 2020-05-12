# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import click
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stan.models import STAN
from stan.preprocessing import BUSIGenerator
from stan.utils.metrics import dice_coef
from stan.utils.losses import *

from .helpers import get_callbacks, get_generators


@click.command()
@click.option('--train_dir', required=True, help='Data directory for training')
@click.option('--val_dir', default=None, help='Data directory for validation')
@click.option('--n_class', default=1, help='Number of classes')
@click.option('--epochs', required=True, help='Number of epochs')
@click.option('--batch_size', default=8, help='Batch size for each iteration')
@click.option('--snapshot-dir', default='.', help='Directory to save snapshot models')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--input_shape', default=256, help='Input shape')
@click.option('--input_channel', default=3)
@click.option('--decode_mode', default='transpose')
@click.option('--tensorboard', default=True, help='Logging with tensorboard')
@click.option('--tensorboard_dir', default='./log', help='Directory to store logging tensorboard files')
@click.option('--model_name', help='Model checkpoint file name')
def train(
    train_dir, val_dir,
    n_class, epochs, batch_size, lr,
    input_shape, input_channel,
    decode_mode,
    snapshot_dir, tensorboard, tensorboard_dir,
    model_name
):
    input_shape = (input_shape, input_shape)

    optimizer = Adam(lr=lr)
    criterion = focal_tversky_loss(gamma=0.75)

    callbacks = get_callbacks(snapshot_dir, model_name, tensorboard, tensorboard_dir, batch_size)

    train_gen, val_gen = get_generators(train_dir, val_dir,
                                        input_shape, input_channel,
                                        horizontal_flip=True)

    model = STAN(
        n_class,
        input_shape=(input_shape[0], input_shape[1], input_channel),
        decode_mode=decode_mode,
        output_activation='sigmoid'
    )
    model.compile(optimizer=optimizer, loss=criterion, metrics=[dice_coef])
    model.fit(train_gen, batch_size=batch_size, 
                epochs=epochs, steps_per_epoch=len(train_gen),
                validation_data=val_gen, validation_steps=1)


if __name__ == "__main__":
    train()
