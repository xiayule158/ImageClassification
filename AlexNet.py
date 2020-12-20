import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from TfUtils import get_data, play_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './alexnet_callbacks'
width, height, channels = 224, 224, 3


class AlexNet:
    def __init__(self):
        pass

    def alexnet(self):
        """
        定义AlexNet的结构
        :return:
        """
        output_num = 3
        net = keras.models.Sequential()

        net.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='same', activation=r'relu',
                                    input_shape=[width, height, channels]))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.MaxPool2D(pool_size=3, strides=2))

        net.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation=r'relu'))
        net.add(keras.layers.MaxPool2D(pool_size=3, strides=2))

        net.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=r'relu'))
        net.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=r'relu'))
        net.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=r'relu'))
        net.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(512, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))

        net.add(keras.layers.Dense(256, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))

        net.add(keras.layers.Dense(output_num, activation=r'softmax'))

        # net.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(0.005, 0.9), metrics=['accuracy'])
        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        net.summary()
        return net

    def deconv_alexnet(self):
        """
        定义AlexNet的结构
        :return:
        """
        output_num = 3
        net = keras.models.Sequential()

        net.add(keras.layers.Conv2D(filters=96, kernel_size=7, strides=2, padding='valid', activation=r'relu',
                                    input_shape=[width, height, channels]))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.MaxPool2D(pool_size=3, strides=2))

        net.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same', activation=r'relu'))
        net.add(keras.layers.MaxPool2D(pool_size=3, strides=2))

        net.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=r'relu'))
        net.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=r'relu'))
        net.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=r'relu'))
        net.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(512, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))

        net.add(keras.layers.Dense(256, activation=r'relu'))
        net.add(keras.layers.Dropout(0.5))

        net.add(keras.layers.Dense(output_num, activation=r'softmax'))

        # net.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(0.005, 0.9), metrics=['accuracy'])
        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        net.summary()
        return net

    def train(self):
        epochs = 100
        model = self.alexnet()
        batch_size = 64
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'alexnet.h5')
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

    def evaluate(self):
        batch_size = 64
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'alexnet.h5')
        model = self.alexnet()
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        model.evaluate(val_generator)


    def get_data_from_log(self):
        train_acc, val_acc = [], []
        train_loss, val_loss = [], []
        with open('./log/alexnet.log', 'r') as f:
            for line in f.readlines():
                rslt = re.findall('- loss: (.*) - accuracy: (.*) - val_loss: (.*) - val_accuracy: (.*)', line)
                if rslt:
                    t_loss, t_acc, v_loss, v_acc = rslt[0]
                    train_loss.append(float(t_loss))
                    train_acc.append(float(t_acc))
                    val_loss.append(float(v_loss))
                    val_acc.append(float(v_acc))
        epochs = list(range(1, len(train_loss)+1))
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/alexnet_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/alexnet_train_val_loss.jpg')


if __name__ == r'__main__':
    # print(tf.__version__)
    alexnet_obj = AlexNet()
    # alexnet_obj.train()
    alexnet_obj.alexnet()
    alexnet_obj.deconv_alexnet()
    # alexnet_obj.evaluate()
    # alexnet_obj.get_data_from_log()

