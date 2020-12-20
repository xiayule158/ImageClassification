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
log_dir = './inception3callbacks'
width, height, channels = 299, 299, 3


class InceptionV3:
    def __init__(self):
        pass

    def inceptionv3_module_v1(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, x, is_max=False, strides=1):
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, strides=strides, padding='same', activation=r'relu')

        p2_conv_1 = keras.layers.Conv2D(filters=n2_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p2_conv_3 = keras.layers.Conv2D(filters=n2_3, kernel_size=3, strides=strides, padding='same', activation=r'relu')

        p3_conv_1 = keras.layers.Conv2D(filters=n3_1, kernel_size=1, strides=strides, padding='same', activation=r'relu')
        p3_conv_3_1 = keras.layers.Conv2D(filters=n3_5, kernel_size=3, strides=1, padding='same', activation=r'relu')
        p3_conv_3_2 = keras.layers.Conv2D(filters=n3_5, kernel_size=3, strides=1, padding='same', activation=r'relu')

        if is_max:
            p4_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=strides, padding='same')
        else:
            p4_maxpool_3 = keras.layers.AvgPool2D(pool_size=3, strides=strides, padding='same')

        p4_conv_1 = keras.layers.Conv2D(filters=n4_1, kernel_size=1, strides=1, padding='same', activation=r'relu')

        p1 = p1_conv_1(x)
        p2 = p2_conv_3(p2_conv_1(x))
        p3 = p3_conv_3_2(p3_conv_3_1(p3_conv_1(x)))
        if n4_1 > 0:
            p4 = p4_conv_1(p4_maxpool_3(x))
        else:
            p4 = p4_maxpool_3(x)

        if n1_1 > 0:
            return keras.layers.concatenate([p1, p2, p3, p4], 3)
        else:
            return keras.layers.concatenate([p2, p3, p4], 3)

    def inceptionv3_module_v2_1(self, n1_1, n1_2, n1_3, n2_1, n2_2, x):
        # 1×1-->3×3-->3×3
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p1_conv_2 = keras.layers.Conv2D(filters=n1_2, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p1_conv_3 = keras.layers.Conv2D(filters=n1_3, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # 1×1-->3×3
        p2_conv_1 = keras.layers.Conv2D(filters=n2_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=n2_2, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # maxpool 3×3
        p3_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')

        p1 = p1_conv_3(p1_conv_2(p1_conv_1(x)))
        p2 = p3_conv_3(p2_conv_1(x))
        p3 = p3_maxpool_3(x)
        return keras.layers.concatenate([p1, p2, p3], 3)

    def inceptionv3_module_v2_2(self, n1_1, n2_2, n3_1, n3_2, n3_3, n4_1, n4_2, n4_3, n4_4, n4_5, x):
        # 1×1
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # ave_pool-->1×1
        p2_pool_1 = keras.layers.AvgPool2D(pool_size=3, strides=1, padding='same')
        p2_conv_2 = keras.layers.Conv2D(filters=n2_2, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # 1×1-->1×7-->7×1
        p3_conv_1 = keras.layers.Conv2D(filters=n3_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_2 = keras.layers.Conv2D(filters=n3_2, kernel_size=(1, 7), strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=n3_3, kernel_size=(7, 1), strides=1, padding='same', activation=r'relu')

        # 1×1-->1×7-->7×1-->1×7-->7×1
        p4_conv_1 = keras.layers.Conv2D(filters=n4_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p4_conv_2 = keras.layers.Conv2D(filters=n4_2, kernel_size=(1, 7), strides=1, padding='same', activation=r'relu')
        p4_conv_3 = keras.layers.Conv2D(filters=n4_3, kernel_size=(7, 1), strides=1, padding='same', activation=r'relu')
        p4_conv_4 = keras.layers.Conv2D(filters=n4_4, kernel_size=(1, 7), strides=1, padding='same', activation=r'relu')
        p4_conv_5 = keras.layers.Conv2D(filters=n4_5, kernel_size=(7, 1), strides=1, padding='same', activation=r'relu')

        p1 = p1_conv_1(x)
        p2 = p2_conv_2(p2_pool_1(x))
        p3 = p3_conv_3(p3_conv_2(p3_conv_1(x)))
        p4 = p4_conv_5(p4_conv_4(p4_conv_3(p4_conv_2(p4_conv_1(x)))))
        return keras.layers.concatenate([p1, p2, p3, p4], 3)

    def inceptionv3_module_v3_1(self, n1_1, n1_2, n1_3, n1_4, n2_1, n2_2, x):
        # 1×1-->1×7-->7×1-->3×3
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p1_conv_2 = keras.layers.Conv2D(filters=n1_2, kernel_size=(1, 7), strides=1, padding='same', activation=r'relu')
        p1_conv_3 = keras.layers.Conv2D(filters=n1_3, kernel_size=(7, 1), strides=1, padding='same', activation=r'relu')
        p1_conv_4 = keras.layers.Conv2D(filters=n1_4, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # 1×1-->3×3
        p2_conv_1 = keras.layers.Conv2D(filters=n2_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=n2_2, kernel_size=3, strides=2, padding='valid', activation=r'relu')

        # maxpool 3×3
        p3_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')

        p1 = p1_conv_4(p1_conv_3(p1_conv_2(p1_conv_1(x))))
        p2 = p3_conv_3(p2_conv_1(x))
        p3 = p3_maxpool_3(x)
        return keras.layers.concatenate([p1, p2, p3], 3)

    def inceptionv3_module_v3_2(self, n1_1, n2_2, n3_1, n3_2, n3_3, n4_1, n4_2, n4_3, n4_4, x):
        # 1×1
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # ave_pool-->3×3
        p2_pool_1 = keras.layers.AvgPool2D(pool_size=3, strides=1, padding='same')
        p2_conv_2 = keras.layers.Conv2D(filters=n2_2, kernel_size=1, strides=1, padding='same', activation=r'relu')

        # 1×1-->1×3,3×1
        p3_conv_1 = keras.layers.Conv2D(filters=n3_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p3_conv_2 = keras.layers.Conv2D(filters=n3_2, kernel_size=(1, 3), strides=1, padding='same', activation=r'relu')
        p3_conv_3 = keras.layers.Conv2D(filters=n3_3, kernel_size=(3, 1), strides=1, padding='same', activation=r'relu')

        # 1×1-->3×3-->7×1-->1×3,3×1
        p4_conv_1 = keras.layers.Conv2D(filters=n4_1, kernel_size=1, strides=1, padding='same', activation=r'relu')
        p4_conv_2 = keras.layers.Conv2D(filters=n4_2, kernel_size=(3, 3), strides=1, padding='same', activation=r'relu')
        p4_conv_3 = keras.layers.Conv2D(filters=n4_3, kernel_size=(1, 3), strides=1, padding='same', activation=r'relu')
        p4_conv_4 = keras.layers.Conv2D(filters=n4_4, kernel_size=(3, 1), strides=1, padding='same', activation=r'relu')

        p1 = p1_conv_1(x)
        p2 = p2_conv_2(p2_pool_1(x))
        p3 = keras.layers.concatenate([p3_conv_2(p3_conv_1(x)), p3_conv_3(p3_conv_1(x))], 3)
        p4 = keras.layers.concatenate([p4_conv_3(p4_conv_2(p4_conv_1(x))), p4_conv_4(p4_conv_2(p4_conv_1(x)))], 3)
        return keras.layers.concatenate([p1, p2, p3, p4], 3)

    def inceptionv3(self):
        output_num = 3
        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation=r'relu')(inpt)
        x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation=r'relu')(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=r'relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(x)

        # 2 layers
        x = keras.layers.Conv2D(filters=80, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, padding='valid', activation='relu')(x)
        x = keras.layers.Conv2D(filters=288, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = keras.layers.BatchNormalization()(x)     # ---->( 35, 35, 288)
        # x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(x)

        # 3 layers
        x = self.inceptionv3_module_v1(n1_1=64, n2_1=96, n2_3=96, n3_1=64, n3_5=96, n4_1=32, x=x)
        x = self.inceptionv3_module_v1(n1_1=64, n2_1=48, n2_3=64, n3_1=96, n3_5=96, n4_1=64, x=x)
        x = self.inceptionv3_module_v1(n1_1=64, n2_1=48, n2_3=64, n3_1=96, n3_5=96, n4_1=64, x=x)
        # x = keras.layers.BatchNormalization()(x)
        # # x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 4 layers
        x = self.inceptionv3_module_v2_1(n1_1=64, n1_2=96, n1_3=96, n2_1=64, n2_2=384, x=x)
        x = self.inceptionv3_module_v2_2(192, 192, 128, 128, 192, 128, 128, 128, 128, 192, x=x)
        x = self.inceptionv3_module_v2_2(192, 192, 160, 160, 192, 160, 160, 160, 160, 192, x=x)
        x = self.inceptionv3_module_v2_2(192, 192, 160, 160, 192, 160, 160, 160, 160, 192, x=x)
        x = self.inceptionv3_module_v2_2(192, 192, 192, 192, 192, 192, 192, 192, 192, 192, x=x)

        # 5 layers
        x = self.inceptionv3_module_v3_1(192, 192, 192, 192, 192, 320, x=x)
        x = self.inceptionv3_module_v3_2(320, 192, 384, 384, 384, 448, 384, 384, 384, x=x)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.AvgPool2D(pool_size=8, strides=8, padding='same')(x)

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

        model = self.inceptionv3()
        batch_size = 64
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'inceptionv3.h5')
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

    def get_data_from_log(self, log_path='./log/inception_v3_log.txt'):
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
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/inception_v3_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/inception_v3_train_val_loss.jpg')
        return train_acc, val_acc, train_loss, val_loss, epochs


if __name__ == r'__main__':
    inceptionv3_obj = InceptionV3()
    # inceptionv3_obj.inceptionv3()
    # inceptionv3_obj.train()
    inceptionv3_obj.get_data_from_log()
