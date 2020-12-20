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
log_dir = './inceptioncallbacks'
width, height, channels = 224, 224, 3


class InceptionBlock(keras.layers.Layer):
    """
    GoogleNet中的Inception类
    """
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, num_outputs=32, **kwargs):
        """
        初始化模块
        :param n1_1: 路径1的输出chanel数
        :param n2_1: 路径2的1×1的输出chanel数
        :param n2_3:路径2的3×3的输出chanel数
        :param n3_1:路径3的1×1的输出chanel数
        :param n3_5:路径3的5×5的输出chanel数
        :param n4_1:路径4的1×1的输出chanel数
        """
        super(InceptionBlock, self).__init__(**kwargs)  # 父类初始化
        self.num_outputs = num_outputs

        self.p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, padding='same', activation=r'relu')
        self.p2_conv_1 = keras.layers.Conv2D(filters=n2_1, kernel_size=1, padding='same', activation=r'relu')
        self.p2_conv_3 = keras.layers.Conv2D(filters=n2_3, kernel_size=3, padding='same', activation=r'relu')
        self.p3_conv_1 = keras.layers.Conv2D(filters=n3_1, kernel_size=1, padding='same', activation=r'relu')
        self.p3_conv_5 = keras.layers.Conv2D(filters=n3_5, kernel_size=5, padding='same', activation=r'relu')
        self.p4_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        self.p4_conv_1 = keras.layers.Conv2D(filters=n4_1, kernel_size=1, padding='same', activation=r'relu')

    def call(self, x):
        """
        前向传播
        :param x:数据变量
        :return:
        """
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_maxpool_3(x))
        return keras.layers.concatenate([p1, p2, p3, p4], 3)

    def get_config(self):
        config = {"num_outputs": self.num_outputs}
        base_config = super(InceptionBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InceptionV1:
    def __init__(self):
        pass

    def inceptionv1_module(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, x):
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, padding='same', activation=r'relu')
        p2_conv_1 = keras.layers.Conv2D(filters=n2_1, kernel_size=1, padding='same', activation=r'relu')
        p2_conv_3 = keras.layers.Conv2D(filters=n2_3, kernel_size=3, padding='same', activation=r'relu')
        p3_conv_1 = keras.layers.Conv2D(filters=n3_1, kernel_size=1, padding='same', activation=r'relu')
        p3_conv_5 = keras.layers.Conv2D(filters=n3_5, kernel_size=5, padding='same', activation=r'relu')
        p4_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        p4_conv_1 = keras.layers.Conv2D(filters=n4_1, kernel_size=1, padding='same', activation=r'relu')

        p1 = p1_conv_1(x)
        p2 = p2_conv_3(p2_conv_1(x))
        p3 = p3_conv_5(p3_conv_1(x))
        p4 = p4_conv_1(p4_maxpool_3(x))
        return keras.layers.concatenate([p1, p2, p3, p4], 3)

    def inceptionv1_model(self):
        output_num = 3

        model = keras.models.Sequential()
        # 1 layers
        model.add(keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu',
                                      input_shape=[width, height, channels]))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        # 2 layers
        model.add(keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        # 3 layers
        model.add(InceptionBlock(n1_1=64, n2_1=96, n2_3=128, n3_1=16, n3_5=32, n4_1=32, num_outputs=256))
        model.add(InceptionBlock(n1_1=128, n2_1=128, n2_3=192, n3_1=32, n3_5=96, n4_1=64, num_outputs=480))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        # 4 layers
        model.add(InceptionBlock(n1_1=192, n2_1=96, n2_3=208, n3_1=16, n3_5=48, n4_1=64, num_outputs=512))
        model.add(InceptionBlock(n1_1=160, n2_1=112, n2_3=224, n3_1=24, n3_5=64, n4_1=64, num_outputs=512))
        model.add(InceptionBlock(n1_1=128, n2_1=128, n2_3=256, n3_1=24, n3_5=64, n4_1=64, num_outputs=512))
        model.add(InceptionBlock(n1_1=112, n2_1=144, n2_3=288, n3_1=32, n3_5=64, n4_1=64, num_outputs=528))
        model.add(InceptionBlock(n1_1=256, n2_1=160, n2_3=320, n3_1=32, n3_5=128, n4_1=128, num_outputs=832))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

        # 5 layers
        model.add(InceptionBlock(n1_1=256, n2_1=160, n2_3=320, n3_1=32, n3_5=128, n4_1=128, num_outputs=832))
        model.add(InceptionBlock(n1_1=384, n2_1=192, n2_3=384, n3_1=48, n3_5=128, n4_1=128, num_outputs=1024))
        model.add(keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same'))

        # 6 layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(output_num, activation=r'softmax'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def inceptionv1(self):
        output_num = 3
        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=r'relu')(inpt)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 2 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 3 layers
        x = self.inceptionv1_module(n1_1=64, n2_1=96, n2_3=128, n3_1=16, n3_5=32, n4_1=32, x=x)
        x = self.inceptionv1_module(n1_1=128, n2_1=128, n2_3=192, n3_1=32, n3_5=96, n4_1=64, x=x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 4 layers
        x = self.inceptionv1_module(n1_1=192, n2_1=96, n2_3=208, n3_1=16, n3_5=48, n4_1=64, x=x)
        x = self.inceptionv1_module(n1_1=160, n2_1=112, n2_3=224, n3_1=24, n3_5=64, n4_1=64, x=x)
        x = self.inceptionv1_module(n1_1=128, n2_1=128, n2_3=256, n3_1=24, n3_5=64, n4_1=64, x=x)
        x = self.inceptionv1_module(n1_1=112, n2_1=144, n2_3=288, n3_1=32, n3_5=64, n4_1=64, x=x)
        x = self.inceptionv1_module(n1_1=256, n2_1=160, n2_3=320, n3_1=32, n3_5=128, n4_1=128, x=x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 5 layers
        x = self.inceptionv1_module(n1_1=256, n2_1=160, n2_3=320, n3_1=32, n3_5=128, n4_1=128, x=x)
        x = self.inceptionv1_module(n1_1=384, n2_1=192, n2_3=384, n3_1=48, n3_5=128, n4_1=128, x=x)
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)

        # 6 layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 1
        np.random.seed(200)

        model = self.inceptionv1()
        batch_size = 32
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'inceptionv1.h5')
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
        with open('./log/inception_v1_log.txt', 'r') as f:
            for line in f.readlines():
                rslt = re.findall('- loss: (.*) - accuracy: (.*) - val_loss: (.*) - val_accuracy: (.*)', line)
                if rslt:
                    t_loss, t_acc, v_loss, v_acc = rslt[0]
                    train_loss.append(float(t_loss))
                    train_acc.append(float(t_acc))
                    val_loss.append(float(v_loss))
                    val_acc.append(float(v_acc))
        # play_curve(epochs, train_acc, './train_acc.jpg')
        # print(np.array(train_acc) > np.array(val_acc))
        epochs = list(range(1, len(train_loss)+1))
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/inception_v1_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/inception_v1_train_val_loss.jpg')


if __name__ == r'__main__':
    inceptionv1_obj = InceptionV1()
    inceptionv1_obj.inceptionv1_model()
    # inceptionv1_obj.train()
    # inceptionv1_obj.get_data_from_log()
