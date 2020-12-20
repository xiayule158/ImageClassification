import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # 设置在CPU上运行
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)     # 内存自增长

# tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')    # 设置GPU可见
# tf.config.experimental.list_logical_devices('GPU')
# tf.config.set_soft_device_placement(True)   # 配合tf.device()使用
# tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)
# ])        # 逻辑切分
# with tf.device(physical_devices[0].name):
#     pass
#     # 设置在那个GPU上运行代码

# MirroredStrategy
# 一机多卡，同步式分布训练，
# CentralStorageStrategy
# MultiWorkerMirroredStrategy
# ParameterServerStrategy


