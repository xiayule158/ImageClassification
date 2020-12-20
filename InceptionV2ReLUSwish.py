"""

"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from TfUtils import get_data
from ReLUSwish import ReLUSwish

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')
for gpu in physical_devices[1:2]:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data'
log_dir = './inceptionv2reluswish_callbacks'
width, height, channels = 224, 224, 3


class InceptionV2:
    def __init__(self):
        pass

    def inceptionv2_module_v1(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, x, is_max=False, strides=1):
        p1_conv_1 = keras.layers.Conv2D(filters=n1_1, kernel_size=1, strides=strides, padding='same')
        p1_ac1 = ReLUSwish()

        p2_conv_1 = keras.layers.Conv2D(filters=n2_1, kernel_size=1, strides=1, padding='same')
        p2_ac1 = ReLUSwish()
        p2_conv_3 = keras.layers.Conv2D(filters=n2_3, kernel_size=3, strides=strides, padding='same')
        p2_ac2 = ReLUSwish()

        p3_conv_1 = keras.layers.Conv2D(filters=n3_1, kernel_size=1, strides=strides, padding='same')
        p3_ac1 = ReLUSwish()
        p3_conv_3_1 = keras.layers.Conv2D(filters=n3_5, kernel_size=3, strides=1, padding='same')
        p3_ac2 = ReLUSwish()
        p3_conv_3_2 = keras.layers.Conv2D(filters=n3_5, kernel_size=3, strides=1, padding='same')
        p3_ac3 = ReLUSwish()

        if is_max:
            p4_maxpool_3 = keras.layers.MaxPool2D(pool_size=3, strides=strides, padding='same')
        else:
            p4_maxpool_3 = keras.layers.AvgPool2D(pool_size=3, strides=strides, padding='same')

        p4_conv_1 = keras.layers.Conv2D(filters=n4_1, kernel_size=1, strides=1, padding='same')
        p4_ac1 = ReLUSwish()

        p1 = p1_ac1(p1_conv_1(x))
        p2 = p2_ac2(p2_conv_3(p2_ac1(p2_conv_1(x))))
        p3 = p3_ac3(p3_conv_3_2(p3_ac2(p3_conv_3_1(p3_ac1(p3_conv_1(x))))))
        if n4_1 > 0:
            p4 = p4_ac1(p4_conv_1(p4_maxpool_3(x)))
        else:
            p4 = p4_maxpool_3(x)

        if n1_1 > 0:
            return keras.layers.concatenate([p1, p2, p3, p4], 3)
        else:
            return keras.layers.concatenate([p2, p3, p4], 3)

    def inceptionv2(self):
        output_num = 3
        inpt = keras.layers.Input(shape=[width, height, channels])

        # 1 layers
        x = keras.layers.SeparableConv2D(filters=64, kernel_size=7, strides=2, padding='same')(inpt)
        x = keras.layers.BatchNormalization()(x)
        x = ReLUSwish()(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 2 layers
        x = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
        x = ReLUSwish()(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(x)
        x = ReLUSwish()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # 3 layers
        x = self.inceptionv2_module_v1(n1_1=64, n2_1=64, n2_3=64, n3_1=64, n3_5=96, n4_1=32, x=x)
        x = self.inceptionv2_module_v1(n1_1=64, n2_1=64, n2_3=96, n3_1=64, n3_5=96, n4_1=64, x=x)
        x = self.inceptionv2_module_v1(n1_1=0, n2_1=128, n2_3=160, n3_1=64, n3_5=96, n4_1=0, x=x, is_max=True, strides=2)

        # 4 layers
        x = self.inceptionv2_module_v1(n1_1=224, n2_1=64, n2_3=96, n3_1=96, n3_5=128, n4_1=128, x=x)
        x = self.inceptionv2_module_v1(n1_1=192, n2_1=96, n2_3=128, n3_1=96, n3_5=128, n4_1=128, x=x)
        x = self.inceptionv2_module_v1(n1_1=128, n2_1=128, n2_3=160, n3_1=128, n3_5=160, n4_1=128, x=x)
        x = self.inceptionv2_module_v1(n1_1=96, n2_1=128, n2_3=160, n3_1=160, n3_5=192, n4_1=128, x=x)
        x = self.inceptionv2_module_v1(n1_1=0, n2_1=128, n2_3=192, n3_1=192, n3_5=256, n4_1=0, x=x, is_max=True, strides=2)

        # 5 layers
        x = self.inceptionv2_module_v1(n1_1=352, n2_1=192, n2_3=320, n3_1=160, n3_5=224, n4_1=128, x=x)
        x = self.inceptionv2_module_v1(n1_1=352, n2_1=192, n2_3=320, n3_1=192, n3_5=224, n4_1=128, x=x, is_max=True)
        x = keras.layers.AvgPool2D(pool_size=7, strides=7, padding='same')(x)

        # 6 layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.7)(x)
        x = keras.layers.Dense(output_num, activation=r'softmax')(x)

        model = keras.Model(inputs=inpt, outputs=x)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        epochs = 100
        np.random.seed(200)

        model = self.inceptionv2()
        batch_size = 128
        train_generator, val_generator = get_data(batch_size=batch_size)
        if not os.path.exists(log_dir):
            os.system('mkdir -p {}'.format(log_dir))
        else:
            os.system('rm -rf {}'.format(log_dir))

        model_path = os.path.join(log_dir, 'inceptionv2_reluswish.h5')
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
    inceptionv2_obj = InceptionV2()
    # inceptionv2_obj.inceptionv2()
    inceptionv2_obj.train()
