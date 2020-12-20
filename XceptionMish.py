"""

"""
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Mish import Mish
from TfUtils import get_data, play_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
for gpu in physical_devices[-1:]:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './xceptionmish_callbacks'
width, height, channels = 299, 299, 3


class Xception:
    def __init__(self):
        pass

    def xceptionv3_entry(self, sc1, sc2, x):
        """

        :param sc1:
        :param sc2:
        :param x:
        :return:
        """
        sc_1 = keras.layers.SeparableConv2D(filters=sc1, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)
        ac1 = Mish()
        bn1 = keras.layers.BatchNormalization()
        sc_2 = keras.layers.SeparableConv2D(filters=sc2, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)
        ac2 = Mish()
        bn2 = keras.layers.BatchNormalization()
        # maxpool 3×3
        mp_3 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        shortcut_conv = keras.layers.Conv2D(filters=sc2, kernel_size=1, strides=2, padding='same')
        p1 = mp_3(ac2(bn2(sc_2(ac1(bn1(sc_1(x)))))))

        return Mish()(p1 + shortcut_conv(x))

    def xceptionv3_middle(self, x):
        sc_1 = keras.layers.SeparableConv2D(filters=728, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)
        ac1 = Mish()
        bn1 = keras.layers.BatchNormalization()

        sc_2 = keras.layers.SeparableConv2D(filters=728, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)
        ac2 = Mish()
        bn2 = keras.layers.BatchNormalization()

        sc_3 = keras.layers.SeparableConv2D(filters=728, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)

        p1 = sc_3(ac2(bn2(sc_2(ac1(bn1(sc_1(x)))))))

        return Mish()(p1 + x)

    def xceptionv3_exit(self, sc1, sc2, x):
        """

        :param sc1:
        :param sc2:
        :param x:
        :return:
        """
        sc_1 = keras.layers.SeparableConv2D(filters=sc1, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)
        ac1 = Mish()
        bn1 = keras.layers.BatchNormalization()

        sc_2 = keras.layers.SeparableConv2D(filters=sc2, kernel_size=3, strides=1,
                                            padding='same',
                                            depth_multiplier=1)
        ac2 = Mish()
        bn2 = keras.layers.BatchNormalization()

        # maxpool 3×3
        mp_3 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        shortcut_conv = keras.layers.Conv2D(filters=sc2, kernel_size=1, strides=2, padding='same')
        p1 = mp_3(ac2(bn2(sc_2(ac1(bn1(sc_1(x)))))))

        return Mish()(p1 + shortcut_conv(x))

    def xception(self):
        output_num = 3
        inpt = keras.layers.Input(shape=[width, height, channels])

        # block 1
        x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='valid')(inpt)
        x = keras.layers.BatchNormalization()(x)
        x = Mish()(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = Mish()(x)

        # block 2
        x = self.xceptionv3_entry(sc1=128, sc2=128, x=x)
        # block 3
        x = self.xceptionv3_entry(sc1=256, sc2=256, x=x)
        # block 4
        x = self.xceptionv3_entry(sc1=728, sc2=728, x=x)

        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)
        x = self.xceptionv3_middle(x)

        x = self.xceptionv3_exit(sc1=728, sc2=1024, x=x)

        x = keras.layers.SeparableConv2D(filters=1536, kernel_size=3, strides=1,
                                         padding='same',
                                         depth_multiplier=1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = Mish()(x)

        x = keras.layers.SeparableConv2D(filters=2048, kernel_size=3, strides=1,
                                         padding='same',
                                         depth_multiplier=1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = Mish()(x)

        x = keras.layers.GlobalAveragePooling2D()(x)

        # 6 layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 100
        np.random.seed(200)

        model = self.xception()
        batch_size = 64
        train_generator, val_generator = get_data(batch_size=batch_size)
        if not os.path.exists(log_dir):
            os.system('mkdir -p {}'.format(log_dir))

        model_path = os.path.join(log_dir, 'xception_mish.h5')
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

    def get_data_from_log(self, log_path='./log/xception_log.txt'):
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
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/xception_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/xception_train_val_loss.jpg')
        return train_acc, val_acc, train_loss, val_loss, epochs


if __name__ == r'__main__':
    xception_obj = Xception()
    # xception_obj.xception()
    xception_obj.train()
    # xception_obj.get_data_from_log()
