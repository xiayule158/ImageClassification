"""

"""

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data, play_curve

tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)     # 内存自增长

width, height, channels = 448, 448, 3


class Yolov1:
    def __init__(self):
        pass

    def conv_block(self, n_filter, n_kernel_s, n_strides, is_pool):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=n_filter, kernel_size=n_kernel_s, strides=n_strides, padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(0.1, ))
        if is_pool:
            model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
        return model

    def yolov1_model(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = keras.models.Sequential()
            model.add(keras.layers.Conv2D(filters=192, kernel_size=7, strides=2, padding='same',
                                          input_shape=[width, height, channels]))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU(0.1, ))
            model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
            model.add(self.conv_block(n_filter=256, n_kernel_s=3, n_strides=1, is_pool=True))
            model.add(self.conv_block(n_filter=128, n_kernel_s=1, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=256, n_kernel_s=3, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=256, n_kernel_s=1, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=512, n_kernel_s=3, n_strides=1, is_pool=True))

            for _ in range(4):
                model.add(self.conv_block(n_filter=256, n_kernel_s=1, n_strides=1, is_pool=False))
                model.add(self.conv_block(n_filter=512, n_kernel_s=3, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=512, n_kernel_s=1, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=1024, n_kernel_s=3, n_strides=1, is_pool=True))

            for _ in range(2):
                model.add(self.conv_block(n_filter=512, n_kernel_s=1, n_strides=1, is_pool=False))
                model.add(self.conv_block(n_filter=1024, n_kernel_s=3, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=1024, n_kernel_s=3, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=1024, n_kernel_s=3, n_strides=2, is_pool=False))

            model.add(self.conv_block(n_filter=1024, n_kernel_s=3, n_strides=1, is_pool=False))
            model.add(self.conv_block(n_filter=1024, n_kernel_s=3, n_strides=1, is_pool=False))

            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(4096, activation=r'relu'))
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(7*7*30))
            model.add(keras.layers.Reshape([7, 7, 30]))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()

        return model


if __name__ == r'__main__':
    yolov1_obj = Yolov1()
    yolov1_obj.yolov1_model()
