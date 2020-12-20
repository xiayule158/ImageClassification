"""

"""
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # 设置在CPU上运行
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data, play_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

image_root = '../data'
log_dir = './mylayer_callbacks'
width, height, channels = 224, 224, 3


class MyLayer(keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=r'relu')
        self.bn = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=r'relu')
        self.conv3 = keras.layers.Conv2D(filters=256 // 4, kernel_size=1, strides=1, padding='same')
        self.output_dim = 56

    def build(self, input_shape):
        self.weight = tf.Variable(tf.random.normal([input_shape[-1], self.output_dim]), name='dense_weight')
        self.bias = tf.Variable(tf.random.normal([self.output_dim]), name='bias_weight')
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        out = self.conv3(self.bn(self.conv2(self.bn(self.conv1(x)))))
        out = keras.layers.Activation('relu')(out + x)
        return out


class MyModel:
    def __init__(self):
        pass

    def my_model(self):
        output_num = 3
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(filters=64, kernel_size=11, strides=4, padding='same', activation=r'relu',
                                      input_shape=[224, 224, 3]))      # ----> 56×56×96
        model.add(MyLayer())
        model.add(keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(output_num, activation=r'relu'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 10
        np.random.seed(200)

        model = self.my_model()
        batch_size = 32
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'mymodel.h5')
        callbacks = [
            keras.callbacks.TensorBoard(log_dir),
            keras.callbacks.ModelCheckpoint(model_path),
        ]
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)

        history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples//batch_size,
                                      epochs=epochs,
                                      validation_data=val_generator,
                                      validation_steps=val_generator.samples//batch_size,
                                      callbacks=callbacks)
        # model.save(model_path)


if __name__ == r'__main__':
    mymodel_obj = MyModel()
    mymodel_obj.train()
    # mymodel_obj.my_model()
