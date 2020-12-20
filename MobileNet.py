import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, Sequential, Model


class MobileNet:

    def mobilenet(self, num_classes, alpha=1, input_shape=(224, 224, 3)):
        model = keras.models.Sequential()

        model.add(
            # layers.Input(input_shape),
            layers.Conv2D(32,
                          (3, 3),
                          strides=2,
                          padding='same',
                          activation='relu', input_shape=input_shape))
        model.add(layers.SeparableConv2D(64,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu')
        )
        model.add(
            layers.SeparableConv2D(128,
                                   (3, 3),
                                   strides=2,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))

        model.add(layers.SeparableConv2D(128,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        )
        model.add(
            layers.SeparableConv2D(256,
                                   (3, 3),
                                   strides=2,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))
        model.add(layers.SeparableConv2D(256,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        )
        model.add(
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=2,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))

        model.add(layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))

        model.add(layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))

        model.add(layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))
        model.add(layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))

        model.add(layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu')
        )
        model.add(
            layers.SeparableConv2D(1024,
                                   (3, 3),
                                   strides=2,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'))

        model.add(layers.SeparableConv2D(1024,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        )
        model.add(layers.AveragePooling2D(pool_size=7, strides=1))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


if __name__ == r'__main__':
    model = MobileNet()
    # model.build((None, 224, 224, 3))
    # model.summary()
    model.mobilenet(3)

