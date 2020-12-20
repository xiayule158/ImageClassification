import os
import cv2
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data, play_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './mobilenetv1_callbacks'
num_class = 3
width, height, channels = 224, 224, 3
batch_size = 8


class MobileNetV1:
    def __init__(self):
        pass

    def mobilenet_v1(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
                                      activation='relu', input_shape=[width, height, channels]))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, strides=2,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=256, kernel_size=3, strides=2,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=256, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=512, kernel_size=3, strides=2,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.SeparableConv2D(filters=512, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=512, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.SeparableConv2D(filters=512, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.SeparableConv2D(filters=512, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.SeparableConv2D(filters=512, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.SeparableConv2D(filters=1024, kernel_size=3, strides=2,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv2D(filters=1024, kernel_size=3, strides=1,
                                               padding='same', activation='relu',
                                               depth_multiplier=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.AveragePooling2D(pool_size=7, strides=1))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(num_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        epochs = 100
        model = self.mobilenet_v1()

        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'mobilenetv1.h5')
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
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'mobilenetv1.h5')
        model = self.mobilenet_v1()
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        model.evaluate(val_generator)
        # 6s 20ms/step - loss: 0.1091 - accuracy: 0.9564

    def get_data_from_log(self):
        train_acc, val_acc = [], []
        train_loss, val_loss = [], []
        with open('./log/mobilev1_log.txt', 'r') as f:
            for line in f.readlines():
                rslt = re.findall('- loss: (.*) - accuracy: (.*) - val_loss: (.*) - val_accuracy: (.*)', line)
                if rslt:
                    t_loss, t_acc, v_loss, v_acc = rslt[0]
                    train_loss.append(float(t_loss))
                    train_acc.append(float(t_acc))
                    val_loss.append(float(v_loss))
                    val_acc.append(float(v_acc))
        epochs = list(range(1, len(train_loss)+1))
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/mobilenetv1_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/mobilenetv1_train_val_loss.jpg')


if __name__ == r'__main__':
    mobile_net_v1 = MobileNetV1()
    # mobile_net_v1.thridparth_mobilenet()
    # mobile_net_v1.train()
    # mobile_net_v1.mobilenet_v1()
    # mobile_net_v1.get_data_from_log()
    mobile_net_v1.evaluate()

