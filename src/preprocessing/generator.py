# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import cv2
import math
import numpy as np
from PIL import Image

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
        self.ids = np.array(range(len(fnames)))
        self.n_samples = len(ids)
        self.resized_shape = resized_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_gen = ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True, **kwargs)
        self.msk_gen = ImageDataGenerator(**kwargs)

    def __len__(self):
        return math.floor(self.n_samples / batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.ids)

    def __getitem__(self, idx):        
        raise NotImplementedError
    
    def _preprocessing(self, img, msk=None):
        raise NotImplementedError

    def _read_data(self, ids):
        imgs, msks = [], []
        for i in ids:
            img = Image.open(os.path.join(self.data_dir, 'images', f'{self.fnames[i]}.png'))
            msk = Image.open(os.path.join(self.data_dir, 'masks', f'{self.fnames[i]}_mask.png'))
            if self.resized_shape:
                img.resize(self.resized_shape, Image.BILINEAR)
                msk.resize(self.resized_shape, Image.BILINEAR)
            imgs.append(np.array(img))
            msks.append(np.array(msk))
        return np.array(imgs), np.array(msks)
