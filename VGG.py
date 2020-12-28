import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './vgg16_callbacks'
width, height, channels = 224, 224, 3


class VGG:
    def __init__(self):
        pass

    def vgg_stack(self, architecture):
        """

        :param architecture: tuple的list
        :return:
        """
        net = []
        for (conv_num, c_n) in architecture:
            for _ in range(conv_num):  # 加入制定的卷积层数
                net.append(keras.layers.Conv2D(filters=c_n, kernel_size=3, padding='same',
                                               activation=r'relu'))
                net.append(keras.layers.BatchNormalization())
                net.append(keras.layers.Activation('relu'))

            net.append(keras.layers.MaxPool2D(strides=2, pool_size=2))
        return net

    def vgg11(self):
        """
        定义VGG11的结构
        :return:
        """
        output_num = 3

        net = keras.models.Sequential()
        # block 1
        net.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=r'relu',
                                    input_shape=[width, height, channels]))
        net.add(keras.layers.MaxPool2D(strides=2, pool_size=2))

        # block 2
        architecture = [(1, 128), (2, 256), (2, 512), (2, 512)]
        layers = self.vgg_stack(architecture)
        for layer in layers:
            net.add(layer)

        # block 3
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(256, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(128, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(output_num, activation=r'softmax'))
        net.summary()

        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return net

    def vgg16(self):
        """
        定义VGG16的结构
        :return:
        """
        output_num = 3
        architecture = [(1, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
        net = keras.models.Sequential()
        net.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=r'relu',
                                    input_shape=[224, 224, 3]))

        layers = self.vgg_stack(architecture)
        for layer in layers:
            net.add(layer)
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(4096, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(256, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))

        net.add(keras.layers.Dense(output_num, activation=r'softmax'))

        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        net.summary()
        return net

    def vgg19(self):
        """
        定义VGG19的结构
        :return:
        """
        output_num = 3
        architecture = [(1, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
        net = keras.models.Sequential()
        net.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=r'relu',
                                    input_shape=[224, 224, 3]))
        layers = self.vgg_stack(architecture)
        for layer in layers:
            net.add(layer)
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(256, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(128, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(output_num, activation=r'softmax'))

        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        net.summary()
        return net

    def train(self):
        epochs = 100
        model = self.vgg16()
        batch_size = 8
        train_generator, val_generator = get_data(batch_size=batch_size)

        model_path = os.path.join(log_dir, 'vgg16.h5')
        callbacks = [
            keras.callbacks.TensorBoard(log_dir),
            keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        ]
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples//batch_size,
                                      epochs=epochs,
                                      validation_data=val_generator,
                                      validation_steps=val_generator.samples//batch_size,
                                      callbacks=callbacks)


if __name__ == r'__main__':
    print(tf.__version__)
    vgg_obj = VGG()
    vgg_obj.vgg11()
    # vgg_obj.train()
