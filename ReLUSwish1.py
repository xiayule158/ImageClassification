import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class ReLUSwish1(Layer):
    """
    Mish Activation Function
    """

    def __init__(self, **kwargs):
        super(ReLUSwish1, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):

        condition = K.greater(inputs, 0)
        return tf.where(condition, inputs, inputs * K.sigmoid(inputs))      # 在0点不可微
        # return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(ReLUSwish1, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        """
        compute_output_shape(self, input_shape)：为了能让Keras内部shape的匹配检查通过，
         这里需要重写compute_output_shape方法去覆盖父类中的同名方法，来保证输出shape是正确的。
         父类Layer中的compute_output_shape方法直接返回的是input_shape这明显是不对的，
         所以需要我们重写这个方法,所以这个方法也是4个要实现的基本方法之一。
        :param input_shape:
        :return:
        """
        return input_shape


if __name__ == r'__main__':
    reluswish1_obj = ReLUSwish1()
