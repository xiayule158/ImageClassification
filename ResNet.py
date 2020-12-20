"""

"""
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
log_dir = './resnet_callbacks'
width, height, channels = 224, 224, 3


class ResNet:
    def __init__(self):
        pass

    def residual_origin_module(self, n1, x):
        """
        原始参差块结构
        :param n1:
        :param n2:
        :param x:
        :return:
        """
        conv1 = keras.layers.Conv2D(filters=n1, kernel_size=3, strides=1, padding='same', activation=r'relu')
        conv2 = keras.layers.Conv2D(filters=n1, kernel_size=3, strides=1, padding='same')
        bn = keras.layers.BatchNormalization()
        out = bn(conv2(conv1(x)))
        return keras.layers.Activation('relu')(out + x)

    def residual_origin_down_module(self, n1, x):
        """
        原始参差块结构
        :param n1:
        :param n2:
        :param x:
        :return:
        """
        conv1 = keras.layers.Conv2D(filters=n1, kernel_size=3, strides=2, padding='same', activation=r'relu')
        conv2 = keras.layers.Conv2D(filters=n1, kernel_size=3, strides=1, padding='same')
        shortcut_conv = keras.layers.Conv2D(filters=n1, kernel_size=1, strides=2, padding='same')

        bn = keras.layers.BatchNormalization()
        out = bn(conv2(conv1(x)))
        return keras.layers.Activation('relu')(out + shortcut_conv(x))

    def residual_a_module(self, n1, n2, n3, x):
        """

        :param n1:
        :param n2:
        :param n3:
        :param x:
        :return:
        """
        conv1 = keras.layers.Conv2D(filters=n1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        bn = keras.layers.BatchNormalization()
        conv2 = keras.layers.Conv2D(filters=n2, kernel_size=3, strides=1, padding='same', activation=r'relu')
        conv3 = keras.layers.Conv2D(filters=n3//4, kernel_size=1, strides=1, padding='same')
        out = conv3(bn(conv2(bn(conv1(x)))))
        # x = conv3(x)

        return keras.layers.Activation('relu')(out + x)

    def residual_b_module(self, n1, n2, n3, x):
        """

        :param n1:
        :param n2:
        :param n3:
        :param x:
        :return:
        """
        conv1 = keras.layers.Conv2D(filters=n1, kernel_size=1, strides=2, padding='same', activation=r'relu')
        conv2 = keras.layers.Conv2D(filters=n2, kernel_size=3, padding='same', activation=r'relu')
        conv3 = keras.layers.Conv2D(filters=n3//4, kernel_size=1, padding='same')
        bn = keras.layers.BatchNormalization()
        conv4 = keras.layers.Conv2D(filters=n3//4, kernel_size=1, strides=2, padding='same')
        out = conv3(bn(conv2(bn(conv1(x)))))
        return keras.layers.Activation('relu')(out + conv4(x))

    def resnet_18(self):
        output_num = 3

        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu')(inpt)
        x = keras.layers.BatchNormalization()(x)

        # 2 layers
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self.residual_origin_module(64, x)
        x = self.residual_origin_module(64, x)

        # 3 layers
        layers_list = [(2, 128), (2, 256), (2, 512)]
        for num_, filters in layers_list:
            x = self.residual_origin_down_module(filters, x)
            for _ in range(num_-1):
                x = self.residual_origin_module(filters, x)

        # 6 layers
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def resnet_34(self):
        output_num = 3

        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu')(inpt)
        x = keras.layers.BatchNormalization()(x)

        # 2 layers
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self.residual_origin_module(64, x)
        x = self.residual_origin_module(64, x)
        x = self.residual_origin_module(64, x)

        # 3 layers
        layers_list = [(4, 128), (6, 256), (3, 512)]
        for num_, filters in layers_list:
            x = self.residual_origin_down_module(filters, x)
            for _ in range(num_-1):
                x = self.residual_origin_module(filters, x)

        # 6 layers
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def resnet_50(self):
        output_num = 3

        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu')(inpt)
        x = keras.layers.BatchNormalization()(x)

        # 2 layers
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self.residual_a_module(64, 64, 256, x)
        x = self.residual_a_module(64, 64, 256, x)
        x = self.residual_a_module(64, 64, 256, x)
        x = keras.layers.BatchNormalization()(x)

        # 3 layers
        x = self.residual_b_module(128, 128, 512, x)
        x = self.residual_a_module(128, 128, 512, x)
        x = self.residual_a_module(128, 128, 512, x)
        x = self.residual_a_module(128, 128, 512, x)
        x = keras.layers.BatchNormalization()(x)

        # 4 layers
        x = self.residual_b_module(256, 256, 1024, x)
        x = self.residual_a_module(256, 256, 1024, x)
        x = self.residual_a_module(256, 256, 1024, x)
        x = self.residual_a_module(256, 256, 1024, x)
        x = self.residual_a_module(256, 256, 1024, x)
        x = self.residual_a_module(256, 256, 1024, x)

        # 5 layers
        x = self.residual_b_module(512, 512, 2048, x)
        x = self.residual_a_module(512, 512, 2048, x)
        x = self.residual_a_module(512, 512, 2048, x)

        # 6 layers
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def resnet_101(self):
        output_num = 3

        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu')(inpt)

        # 2 layers
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self.residual_a_module(64, 64, 256, x)
        x = self.residual_a_module(64, 64, 256, x)
        x = self.residual_a_module(64, 64, 256, x)

        # 3 ×4=1+3
        x = self.residual_b_module(128, 128, 512, x)
        for i in range(3):
            x = self.residual_a_module(128, 128, 512, x)

        # 4 ×23=1+22
        x = self.residual_b_module(256, 256, 1024, x)
        for i in range(22):
            x = self.residual_a_module(256, 256, 1024, x)

        # 5 3=1+2
        x = self.residual_b_module(512, 512, 2048, x)
        for i in range(2):
            x = self.residual_a_module(512, 512, 2048, x)

        # 6 layers
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def resnet_152(self):
        output_num = 3
        layers_filters_list = [[8, (128, 128, 512)], [36, (256, 256, 1024)], [3, (512, 512, 2048)]]
        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu')(inpt)
        x = keras.layers.BatchNormalization()(x)

        # 2 layers
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self.residual_a_module(64, 64, 256, x)
        x = self.residual_a_module(64, 64, 256, x)
        x = self.residual_a_module(64, 64, 256, x)

        for layer, filter_tuple in layers_filters_list:
            x = self.residual_b_module(filter_tuple[0], filter_tuple[1], filter_tuple[2], x)
            for _ in range(layer - 1):
                x = self.residual_a_module(filter_tuple[0], filter_tuple[1], filter_tuple[2], x)

        # 6 layers
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 100
        np.random.seed(200)

        model = self.resnet_101()
        batch_size = 64
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'resnet101.h5')
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
        # model.save(model_path)

    def get_data_from_log(self):
        train_acc, val_acc = [], []
        train_loss, val_loss = [], []
        with open('./log/resnet101_log.txt', 'r') as f:
            for line in f.readlines():
                rslt = re.findall('- loss: (.*) - accuracy: (.*) - val_loss: (.*) - val_accuracy: (.*)', line)
                if rslt:
                    t_loss, t_acc, v_loss, v_acc = rslt[0]
                    train_loss.append(float(t_loss))
                    train_acc.append(float(t_acc))
                    val_loss.append(float(v_loss))
                    val_acc.append(float(v_acc))
        epochs = list(range(1, len(train_loss)+1))
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/resnet101_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/resnet101_train_val_loss.jpg')


if __name__ == r'__main__':
    resnet_obj = ResNet()
    resnet_obj.resnet_18()
    # resnet_obj.train()
    # resnet_obj.get_data_from_log()

