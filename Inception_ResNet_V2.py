"""

"""
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data, play_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
image_root = '../data'
log_dir = './inception_resnet2_callbacks'
width, height, channels = 299, 299, 3


class InceptionResnetV2:
    def __init__(self):
        pass

    def stem_module(self, x):
        """
        stem模块
        :param x:
        :return:
        """
        x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation=r'relu')(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation=r'relu')(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=r'relu')(x)
        pool1_1 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')
        conv1_1 = keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding='valid', activation=r'relu')
        x = keras.layers.concatenate([pool1_1(x), conv1_1(x)], 3)

        conv1_1 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=r'relu')
        conv1_2 = keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='valid', activation=r'relu')

        conv2_1 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=r'relu')
        conv2_2 = keras.layers.Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same', activation=r'relu')
        conv2_3 = keras.layers.Conv2D(filters=64, kernel_size=(1, 7), strides=1, padding='same', activation=r'relu')
        conv2_4 = keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='valid', activation=r'relu')
        x = keras.layers.concatenate([conv1_2(conv1_1(x)), conv2_4(conv2_3(conv2_2(conv2_1(x))))], 3)

        pool1_1 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')
        conv1_1 = keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, padding='valid', activation=r'relu')
        return keras.layers.concatenate([pool1_1(x), conv1_1(x)], 3)

    def inception_res_A_module(self, x):
        # branch2
        p2_conv_1 = keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # branch3
        p3_conv_1 = keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation=r'relu')

        # branch4
        p4_conv_1 = keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p4_conv_2 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation=r'relu')
        p4_conv_3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=r'relu')

        p2 = p2_conv_1(x)
        p3 = p3_conv_3(p3_conv_1(x))
        p4 = p4_conv_3(p4_conv_2(p4_conv_1(x)))
        x_2 = keras.layers.concatenate([p2, p3, p4], 3)

        x_2 = keras.layers.Conv2D(filters=384, kernel_size=1, strides=1, padding='same', activation=r'relu')(x_2)
        p1 = x

        return keras.layers.Activation('relu')(p1 + x_2)

    def reduction_A_module(self, x):
        # maxpool 3×3
        p1_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')

        # 3×3
        p2_conv_1 = keras.layers.Conv2D(filters=384, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # 1×1-->3×3-->3×3
        p3_conv_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_2 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=384, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        p1 = p1_maxpool_3(x)
        p2 = p2_conv_1(x)
        p3 = p3_conv_3(p3_conv_2(p3_conv_1(x)))

        return keras.layers.concatenate([p1, p2, p3], 3)

    def inception_res_B_module(self, x):
        # branch2
        p2_conv_1 = keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # branch3
        p3_conv_1 = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_2 = keras.layers.Conv2D(filters=160, kernel_size=(1, 7), strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=192, kernel_size=(7, 1), strides=1, padding='same', activation=r'relu')

        p1 = x
        p2 = p2_conv_1(x)
        p3 = p3_conv_3(p3_conv_2(p3_conv_1(x)))
        x_2 = keras.layers.concatenate([p2, p3], 3)

        x_2 = keras.layers.Conv2D(filters=1152, kernel_size=1, strides=1, padding='same', activation=r'relu')(x_2)

        return keras.layers.Activation('relu')(p1 + x_2)

    def reduction_B_module(self, x):
        # maxpool 3×3
        p1_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')

        # 3×3
        p2_conv_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p2_conv_2 = keras.layers.Conv2D(filters=384, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # 1×1-->3×3
        p3_conv_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_2 = keras.layers.Conv2D(filters=288, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # 1×1-->3×3-->3×3
        p4_conv_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p4_conv_2 = keras.layers.Conv2D(filters=288, kernel_size=3, strides=1, padding='same', activation=r'relu')
        p4_conv_3 = keras.layers.Conv2D(filters=320, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        p1 = p1_maxpool_3(x)
        p2 = p2_conv_2(p2_conv_1(x))
        p3 = p3_conv_2(p3_conv_1(x))
        p4 = p4_conv_3(p4_conv_2(p4_conv_1(x)))

        return keras.layers.concatenate([p1, p2, p3, p4], 3)

    def inception_res_C_module(self, x):
        # branch1

        # branch2
        p2_conv_1 = keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # branch3
        p3_conv_1 = keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_2 = keras.layers.Conv2D(filters=224, kernel_size=(1, 3), strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=256, kernel_size=(3, 1), strides=1, padding='same', activation=r'relu')

        p1 = x
        p2 = p2_conv_1(x)
        p3 = p3_conv_3(p3_conv_2(p3_conv_1(x)))
        x_2 = keras.layers.concatenate([p2, p3], 3)

        x_2 = keras.layers.Conv2D(filters=2144, kernel_size=1, strides=1, padding='same', activation=r'relu')(x_2)

        return keras.layers.Activation('relu')(p1 + x_2)

    def inception_res_v2(self):
        output_num = 3
        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = self.stem_module(inpt)
        x = keras.layers.BatchNormalization()(x)

        # 2 5×Inception-A
        x = self.inception_res_A_module(x)
        x = self.inception_res_A_module(x)
        x = self.inception_res_A_module(x)
        x = self.inception_res_A_module(x)
        x = self.inception_res_A_module(x)
        x = keras.layers.BatchNormalization()(x)

        # 3 Reduction-A
        x = self.reduction_A_module(x)

        # 4 10×Inception-B
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)
        x = self.inception_res_B_module(x)

        # 5 Reduction-B
        x = self.reduction_B_module(x)

        # 6 5×Inception-C
        x = self.inception_res_C_module(x)
        x = self.inception_res_C_module(x)
        x = self.inception_res_C_module(x)
        x = self.inception_res_C_module(x)
        x = self.inception_res_C_module(x)

        #
        x = keras.layers.AvgPool2D(pool_size=8, strides=8, padding='same')(x)

        # 6 layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 100
        np.random.seed(200)

        model = self.inception_res_v2()
        batch_size = 32
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'inception_res_v2.h5')
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

    def get_data_from_log(self, log_path='./log/inception_res_v2_log.txt'):
        train_acc, val_acc = [], []
        train_loss, val_loss = [], []
        with open(log_path, 'r') as f:
            for line in f.readlines():
                rslt = re.findall('- loss: (.*) - accuracy: (.*) - val_loss: (.*) - val_accuracy: (.*)', line)
                if rslt:
                    t_loss, t_acc, v_loss, v_acc = rslt[0]
                    train_loss.append(float(t_loss))
                    train_acc.append(float(t_acc))
                    val_loss.append(float(v_loss))
                    val_acc.append(float(v_acc))
        epochs = list(range(1, len(train_loss)+1))
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/inception_res_v2_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/inception_res_v2_train_val_loss.jpg')
        return train_acc, val_acc, train_loss, val_loss, epochs


if __name__ == r'__main__':
    inception_res_obj = InceptionResnetV2()
    # inception_res_obj.inception_res_v2()
    # inception_res_obj.train()
    inception_res_obj.get_data_from_log()
