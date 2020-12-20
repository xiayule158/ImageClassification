"""
卷积理解
"""
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # 设置在CPU上运行
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
image_root = '../data/Training/cats'


def show_conv():
    np.random.seed(1671)
    sobel_x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobel_y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    img = cv2.imread(os.path.join(image_root, 'cat.1.jpg'), 0)

    cv2.imshow('src', img)
    cv2.waitKey()
    input_array = img.reshape([1, img.shape[0], img.shape[1], 1])
    input_array = np.array(input_array, dtype=np.float64)
    model = keras.models.Sequential()
    conv1 = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=keras.initializers.Constant(value=sobel_x),
                                input_shape=input_array.shape[1:])
    conv2 = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=keras.initializers.Constant(value=sobel_y),
                                input_shape=input_array.shape[1:])
    model.add(conv1)
    model.summary()
    out_put = conv1(input_array)
    print(out_put.shape)
    cv2.imshow('x', np.array(out_put)[0, :, :, :])
    cv2.waitKey()

    deconv1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same',
                                           kernel_initializer=keras.initializers.Constant(
                                               value=np.linalg.pinv(sobel_x)),
                                           input_shape=np.array(out_put).shape[1:])
    deconv_x = deconv1(np.array(out_put))
    cv2.imshow('deconv_x', np.array(deconv_x)[0, :, :, :])
    cv2.waitKey()

    out_put = conv2(input_array)
    print(out_put.shape)
    cv2.imshow('y', np.array(out_put)[0, :, :, :])
    cv2.waitKey()


def spatial_attention():
    img = cv2.imread(os.path.join(image_root, 'cat.1.jpg'))
    cv2.imshow('1', img)
    input_array = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
    input_array = np.array(input_array, dtype=np.float64)
    avg_out = tf.reduce_mean(input_array, 3)
    max_out = tf.reduce_max(input_array, 3)

    ave_img = np.array(avg_out, dtype=np.uint8).reshape(img.shape[0], img.shape[1], 1)
    cv2.imshow('ave', ave_img)

    max_img = np.array(max_out, dtype=np.uint8).reshape(img.shape[0], img.shape[1], 1)
    cv2.imshow('max', max_img)
    cv2.waitKey()
    # plt.imshow(show_img)
    # cv2.waitKey()


def conv_calculate():
    np.random.seed(1671)
    sobel_x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]*3
    sobel_y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]*3
    img = np.array(range(27))
    input_array = img.reshape([1, 3, 3, 3])
    input_array = np.array(input_array, dtype=np.float64)
    model = keras.models.Sequential()
    conv1 = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='valid',
                                kernel_initializer=keras.initializers.Constant(value=sobel_x),
                                input_shape=input_array.shape[1:])
    conv2 = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=keras.initializers.Constant(value=sobel_y),
                                input_shape=input_array.shape[1:])
    model.add(conv1)
    model.summary()
    out_put = conv1(input_array)
    print(out_put.shape)
    print(np.array(out_put)[0, :, :, :])

    out_put = conv2(input_array)
    print(out_put.shape)
    # cv2.imshow('2', np.array(out_put)[0, :, :, :])
    # cv2.waitKey()


def map_a():
    img = cv2.imread('../data/Training/cats/cat.10.jpg', 0)
    cv2.imshow('1', img)
    # cv2.waitKey()

    img = img // 2
    cv2.imshow('2', img)
    cv2.waitKey()


if __name__ == r'__main__':
    # print(tf.__version__)
    show_conv()
    # conv_calculate()
    # spatial_attention()
    # map_a()
