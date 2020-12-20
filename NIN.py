"""
Network in Network
"""
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data, play_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
for gpu in physical_devices[-1:]:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './nin_callbacks'
width, height, channels = 224, 224, 3


class NIN:
    def __init__(self):
        pass

    def mlpconv(self,  f_num, k_size, s, pad, x):

        conv1 = keras.layers.Conv2D(filters=f_num, kernel_size=k_size, strides=s, padding=pad)
        ac1 = keras.layers.Activation('relu')
        bn1 = keras.layers.BatchNormalization()

        conv2 = keras.layers.Conv2D(filters=f_num, kernel_size=1, strides=1, padding=pad)
        ac2 = keras.layers.Activation('relu')
        bn2 = keras.layers.BatchNormalization()

        conv3 = keras.layers.Conv2D(filters=f_num, kernel_size=1, strides=1, padding=pad)
        ac3 = keras.layers.Activation('relu')
        bn3 = keras.layers.BatchNormalization()

        # return conv3(conv2(conv1(x)))
        return ac3(bn3(conv3(ac2(bn2(conv2(ac1(bn1(conv1(x)))))))))

    def nin(self):
        output_num = 3
        inpt = keras.layers.Input(shape=[width, height, channels])

        # block 1
        x = self.mlpconv(96, 11, 4, 'valid', inpt)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = self.mlpconv(256, 5, 1, 'same', x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = self.mlpconv(384, 3, 1, 'same', x)
        x = self.mlpconv(512, 3, 1, 'same', x)
        x = self.mlpconv(1024, 3, 1, 'same', x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

        # 6 layers
        x = self.mlpconv(output_num, 3, 1, 'same', x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 100
        np.random.seed(200)

        model = self.nin()
        batch_size = 128
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'nin.h5')
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

    def get_data_from_log(self, log_path='./log/nin_log.txt'):
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
        play_curve(epochs, train_acc, val_acc, fig_path='./train_val_plot/nin_train_val_acc.jpg')
        play_curve(epochs, train_loss, val_loss, acc=False, fig_path='./train_val_plot/nin_train_val_loss.jpg')
        return train_acc, val_acc, train_loss, val_loss, epochs


if __name__ == r'__main__':
    nin_obj = NIN()
    # nin_obj.nin()
    nin_obj.train()
    # nin_obj.get_data_from_log()
