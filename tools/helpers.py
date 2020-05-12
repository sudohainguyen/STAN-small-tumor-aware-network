# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)

from stan.preprocessing import BUSIGenerator


def get_callbacks(
    snapshot_dir,
    model_name,
    use_tensorboard, tensorboard_dir,
    batch_size
):
    """Get callbacks for fitting models

    Arguments:
        snapshot_dir {str} -- Directory to save snapshot models
        model_name {str} -- Model's name
        use_tensorboard {boolean} -- Logging with tensorboard
        tensorboard_dir {str} -- Directory to store logging tensorboard files
        batch_size {int} -- Batch size for each iteration

    Returns:
        list -- List of callbacks
    """    
    callbacks = [
        ModelCheckpoint(os.path.join(snapshot_dir, f'{model_name}.h5'),
                        monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=8),
        ReduceLROnPlateau(monitor='val_loss', patience=3)
    ]

    if use_tensorboard:
        os.makedirs(tensorboard_dir)
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=False,
            write_images=True,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    return callbacks


def get_generators(
    train_dir, val_dir,
    input_shape, input_channel,
    **kwargs
):
    """Get training generator and validation generator

    Arguments:
        train_dir {str} -- Data directory for training
        val_dir {str} -- Data directory for validation
        input_shape {tuple} -- Size of input images
        input_channel {int} -- 1 or 3, number channels of images

    Returns:
        tuple -- training gen, validation gen
    """    
    train_fnames = [os.path.splitext(f)[0] 
                    for f in os.listdir(os.path.join(train_dir, 'images'))]
    train_gen = BUSIGenerator(
        train_fnames, train_dir,
        resized_shape=input_shape,
        input_channel=input_channel,
        **kwargs,
        # horizontal_flip=True,
        # rotation_range=20, width_shift_range=10,
    )
    val_gen = None
    if val_dir:
        val_fnames = [os.path.splitext(f)[0] 
                        for f in os.listdir(os.path.join(val_dir, 'images'))]
        val_gen = BUSIGenerator(
            val_fnames, val_dir,
            resized_shape=input_shape,
            input_channel=input_channel,
            **kwargs,
        )

    return train_gen, val_gen
