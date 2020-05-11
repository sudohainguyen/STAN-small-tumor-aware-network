# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import click
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from stan.model import STAN
from preprocessing import BUSIGenerator
from utils.metrics import dice_coef
from utils.losses import *


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
def train(
    train_dir,
    val_dir,
    n_class,
    epochs,
    batch_size,
    snapshot_dir,
    lr,
    input_shape,
    input_channel,
    decode_mode
):
    input_shape = (input_shape, input_shape)

    optimizer = Adam(lr=lr)
    criterion = focal_tversky_loss(gamma=0.75)

    callbacks = [
        ModelCheckpoint(snapshot_dir, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=8),
        ReduceLROnPlateau(monitor='val_loss', patience=3)
    ]

    train_fnames = [os.path.splitext(f)[0] 
                    for f in os.listdir(os.path.join(train_dir, 'images'))]
    train_gen = BUSIGenerator(
        train_dir,
        resized_shape=input_shape,
        input_channel=input_channel,
        horizontal_flip=True,
        # rotation_range=20, width_shift_range=10,
    )

    val_gen = None
    if val_dir:
        val_fnames = [os.path.splitext(f)[0] 
                        for f in os.listdir(os.path.join(val_dir, 'images'))]
        val_gen = BUSIGenerator(
            val_dir,
            resized_shape=input_shape,
            input_channel=input_channel
        )

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
