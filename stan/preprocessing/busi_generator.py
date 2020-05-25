# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import cv2
import numpy as np

from .generator import BaseGenerator


class BUSIGenerator(BaseGenerator):
    def __init__(self, fnames, input_channel=3, **kwargs):
        super(BUSIGenerator, self).__init__(fnames, **kwargs)
        self.input_channel = input_channel

    def _preprocessing(self, imgs, msks=None):
        # imgs[..., 0] -= 103.939
        # imgs[..., 1] -= 116.779
        # imgs[..., 2] -= 123.68
        # imgs -= 1.
        imgs /= 255
        imgs = self.img_gen.random_transform(x=imgs, seed=self.seed)
        if msks is not None:
            msks /= 255
            msks = self.msk_gen.random_transform(x=msks, seed=self.seed)
            return imgs, msks
        return imgs

    def _read_data(self, ids):
        imgs = np.empty((self.batch_size, self.resized_shape[0],
                        self.resized_shape[1], self.input_channel))
        msks = np.empty((self.batch_size, self.resized_shape[0],
                         self.resized_shape[1], 1))

        for i, index in enumerate(ids):
            file_name = self.fnames[index]
            read_im_mode = 1
            if self.input_channel == 1:
                read_im_mode = 0
            img = cv2.imread(
                os.path.join(self.data_dir, 'images', f'{file_name}.png'),
                read_im_mode
            )
            msk = cv2.imread(
                os.path.join(self.data_dir, 'masks', f'{file_name}.png'),
                read_im_mode
            )

            if self.resized_shape:
                img = cv2.resize(img, self.resized_shape)
                msk = cv2.resize(msk, self.resized_shape)

            imgs[i] = img
            msks[i] = np.expand_dims(msk, axis=2)
        
        return imgs, msks
