"""

"""
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from TfUtils import get_data, play_curve
from resnet import ResNet18, ResNet34, ResNet50

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
image_root = '../data'
log_dir = './resnetcbam_callbacks'
width, height, channels = 224, 224, 3


class ResNet:
    def __init__(self):
        pass

    def resnet50_cbam(self):
        model = ResNet18()
        model.build(input_shape=(None, width, height, channels))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 2
        np.random.seed(200)

        model = self.resnet50_cbam()
        batch_size = 1
        train_generator, val_generator = get_data(batch_size=batch_size)
        model_path = os.path.join(log_dir, 'resnet50_cbam.h5')
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


if __name__ == r'__main__':
    resnet_obj = ResNet()
    # resnet_obj.resnet50_cbam()
    resnet_obj.train()
    # resnet_obj.get_data_from_log()