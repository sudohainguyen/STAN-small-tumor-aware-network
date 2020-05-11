# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, Iterator


class BaseGenerator(Iterator):
    def __init__(
        self,
        fnames,
        data_dir='.',
        n_classes=1,
        resized_shape=(256, 256),
        batch_size=8,
        shuffle=False,
        seed=None,
        **kwargs
    ):
        """Generator initialization, inherits Iterator class

        Arguments:
            fnames {list} -- List of filenames from dataset (not including extension part)

        Keyword Arguments:
            data_dir {str} -- The main dataset directory (default: {'.'})
            n_classes {int} -- Number of classes (default: {1})
            resized_shape {tuple} -- All images are reshaped to this size (default: {(256, 256)})
            batch_size {int} -- Batch size (default: {8})
            shuffle {bool} -- (default: {False})
        """    
        self.fnames = fnames
        self.data_dir = data_dir
        self.seed = seed
        self.ids = np.array(range(len(fnames)))
        self.n_samples = len(self.ids)
        self.resized_shape = resized_shape
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.cursor = 0

        self.img_gen = ImageDataGenerator(**kwargs)
        self.msk_gen = ImageDataGenerator(**kwargs)

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)
    
    def on_epoch_end(self):
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.ids)

    def __getitem__(self, idx):
        indices = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs, msks = self._read_data(indices)
        imgs, msks = self._preprocessing(imgs, msks)
        return imgs, msks

    def _preprocessing(self, imgs, msks=None):
        raise NotImplementedError

    def _read_data(self, ids):
        raise NotImplementedError
