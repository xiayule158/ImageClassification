import os
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
for gpu in physical_devices[0:1]:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './alexnetrelu_callbacks'
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

        net.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='same',
                                    input_shape=[width, height, channels]))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.Activation('relu'))
        net.add(keras.layers.MaxPool2D(pool_size=3, strides=2))

        net.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same'))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.Activation('relu'))
        net.add(keras.layers.MaxPool2D(pool_size=3, strides=2))

        net.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same'))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.Activation('relu'))

        net.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same'))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.Activation('relu'))

        net.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'))
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.Activation('relu'))

        net.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(1024))
        net.add(keras.layers.Activation('relu'))

        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(512))
        net.add(keras.layers.Activation('relu'))

        net.add(keras.layers.Dropout(0.5))
        net.add(keras.layers.Dense(output_num, activation=r'softmax'))

        # net.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(0.005, 0.9), metrics=['accuracy'])
        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        net.summary()
        return net

    def train(self):
        epochs = 500
        model = self.alexnet()
        batch_size = 220
        train_generator, val_generator = get_data(batch_size=batch_size)
        if not os.path.exists(log_dir):
            os.system('mkdir -p {}'.format(log_dir))
        else:
            os.system('rm -rf {}'.format(log_dir))
        model_path = os.path.join(log_dir, 'alexnet_relu.h5')
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
        model_path = os.path.join(log_dir, 'alexnet_relu.h5')
        model = self.alexnet()
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        model.evaluate(val_generator)


if __name__ == r'__main__':
    alexnet_obj = AlexNet()
    alexnet_obj.train()
    # alexnet_obj.alexnet()
    # alexnet_obj.evaluate()

