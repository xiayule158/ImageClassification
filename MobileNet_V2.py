import os
import re
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data, play_curve
from MobileNetV2 import mobilenetv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './mobilenetv2_callbacks'
num_class = 3
width, height, channels = 224, 224, 3
batch_size = 64


class MobileNetV2:
    def __init__(self):
        pass

    def ReLU6(self):
        return keras.layers.Lambda(lambda x: tf.nn.relu6(x))

    def linearbottleneck(self, t, in_channel, out_channel, strides, x):
        residual = keras.models.Sequential([
            keras.layers.Conv2D(in_channel * t, kernel_size=1, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            self.ReLU6(),
            keras.layers.DepthwiseConv2D((3, 3), strides=strides, padding='same'),
            keras.layers.BatchNormalization(),
            self.ReLU6(),
            keras.layers.Conv2D(filters=out_channel, kernel_size=1, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
        ])
        out = residual(x)
        if strides == 1 and in_channel == out_channel:
            out = out + x
        return out

    def stack_bottleneck(self, n, t, in_channel, out_channel, strides, x):

        x = self.linearbottleneck(t, in_channel, out_channel, strides, x)
        while n - 1:
            x = self.linearbottleneck(t, in_channel, out_channel, 1, x)
            n -= 1
        return x

    def mobilenet_v2(self):
        inpt = keras.layers.Input(shape=[width, height, channels])
        # head
        x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
                                input_shape=[width, height, channels])(inpt)
        x = keras.layers.BatchNormalization()(x)
        x = self.ReLU6()(x)

        x = self.linearbottleneck(1, 32, 16, 1, x)
        x = self.stack_bottleneck(2, 6, 16, 24, 2, x)
        x = self.stack_bottleneck(3, 6, 24, 32, 2, x)
        x = self.stack_bottleneck(4, 6, 32, 64, 2, x)
        x = self.stack_bottleneck(3, 6, 64, 96, 1, x)
        x = self.stack_bottleneck(3, 6, 96, 160, 2, x)
        x = self.linearbottleneck(6, 160, 320, 1, x)

        x = keras.layers.Conv2D(filters=1280, kernel_size=1, strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = self.ReLU6()(x)

        x = keras.layers.AveragePooling2D(pool_size=7, strides=1)(x)
        x = keras.layers.Conv2D(filters=num_class, kernel_size=1, strides=1, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(num_class, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 100
        model = self.mobilenet_v2()

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
        model = self.mobilenet_v2()
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        model.evaluate(val_generator)

    def get_data_from_log(self):
        train_acc, val_acc = [], []
        train_loss, val_loss = [], []
        with open('./log/mobilenetv2_log.txt', 'r') as f:
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
    mobile_net_v2 = MobileNetV2()
    # mobile_net_v2.train()
    # mobile_net_v2.mobilenet_v2()
    # mobile_net_v2.get_data_from_log()
    mobile_net_v2.evaluate()

