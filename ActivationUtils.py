"""
激活函数工具函数
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K


def get_acc_loss(log_path):
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
    epochs = list(range(1, len(train_loss) + 1))
    return train_acc, val_acc, train_loss, val_loss, epochs


def plot_curve(train_acc1, val_acc1, train_acc2, val_acc2, epochs):
    iter_list = np.array(epochs)
    plt.rcParams['figure.figsize'] = (10, 7)  # 图片尺寸
    # fig = plt.figure(figsize=(10, 7), dpi=700)  # defaults to rc figure.dpi)
    lfw1, = plt.plot(iter_list, train_acc1, color='r')
    cfd1, = plt.plot(iter_list, val_acc1, color='b', linestyle='--')
    lfw2, = plt.plot(iter_list, train_acc2, color='r', marker='D',
                     markerfacecolor='none')
    cfd2, = plt.plot(iter_list, val_acc2, color='b', linestyle='dashed', marker='D',
                     markerfacecolor='none')

    # 6.添加legend
    plt.legend(handles=[lfw1, cfd1, lfw2, cfd2],
               labels=['train_acc1', 'val_acc1', 'train_acc2', 'val_acc2'],
               loc='lower right', fontsize=20)
    # plt.ylim(0.90, 1.0)
    # plt.xlim(80, None)
    plt.tick_params(labelsize=20)  # 设置x,y轴字体大小
    plt.xlabel('iter times/k', fontsize=25)
    plt.ylabel('val acc/%', fontsize=25)
    plt.tight_layout()  # 去除白边
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_ylim(0.5, 1.0)

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    # plt.savefig('../../baiduface/train_pic/{}1.svg'.format('1'), dpi=700, )
    # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
    # 最好为矢量图（eps、AI、pdf、Visco、emf、dxf等）
    plt.show()


def compare_data(plot_dict, epochs, save_path=''):
    """

    :param plot_dict: {'label':[array, color]}
    :param epochs:
    :return:
    """
    iter_list = np.array(epochs)
    plt.rcParams['figure.figsize'] = (10, 7)  # 图片尺寸
    # fig = plt.figure(figsize=(10, 7), dpi=700)  # defaults to rc figure.dpi)

    curve_list = []
    for k, curve_color in plot_dict.items():
        curve, = plt.plot(iter_list, curve_color[0], color=curve_color[1])
        curve_list.append([curve, k])

    # 2.添加legend
    plt.legend(handles=[item[0] for item in curve_list],
               labels=[item[1] for item in curve_list],
               loc='lower right', fontsize=20)
    # plt.ylim(0.90, 1.0)
    # plt.xlim(80, None)
    plt.tick_params(labelsize=20)  # 设置x,y轴字体大小
    plt.xlabel('iter times/k', fontsize=25)
    plt.ylabel('val acc/%', fontsize=25)
    plt.tight_layout()  # 去除白边
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_ylim(0.5, 1.0)

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    #
    # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
    # 最好为矢量图（eps、AI、pdf、Visco、emf、dxf等）
    if save_path:
        plt.savefig(save_path, dpi=700, )
    plt.show()


def XceptionCompare():
    train_acc1, val_acc1, train_loss1, val_loss1, epochs = get_acc_loss('./log/XceptionMish_log.txt')
    train_acc2, val_acc2, train_loss2, val_loss2, epochs = get_acc_loss('./log/XceptionReLU_log.txt')

    train_acc3, val_acc3, train_loss3, val_loss3, epochs = get_acc_loss('./log/XceptionReLUSwish_log.txt')
    train_acc4, val_acc4, train_loss4, val_loss4, epochs = get_acc_loss('./log/XceptionSwish_log.txt')

    # 1.比较TrainingAcc
    plot_dict = {'MishTrainAcc': [train_acc1, 'r'], 'ReLUTrainAcc': [train_acc2, 'b'],
                 'ReLUSwishTrainAcc': [train_acc3, 'g'], 'SwishTrainAcc': [train_acc4, 'y']}
    mean_train_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_train_acc = sorted(mean_train_acc, key=lambda x: x[1])
    print('XceptionCompareResult:')
    print(mean_train_acc)
    compare_data(plot_dict, epochs)

    # 2.比较ValAcc
    plot_dict = {'MishValAcc': [val_acc1, 'r'], 'ReLUValAcc': [val_acc2, 'b'],
                 'ReLUSwishValAcc': [val_acc3, 'g'], 'SwishValAcc': [val_acc4, 'y']}
    mean_val_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_val_acc = sorted(mean_val_acc, key=lambda x: x[1])
    print(mean_val_acc)
    compare_data(plot_dict, epochs)


def AlexNetCompare():
    train_acc1, val_acc1, train_loss1, val_loss1, epochs = get_acc_loss('./log/AlexnetMish_log.txt')
    train_acc2, val_acc2, train_loss2, val_loss2, epochs = get_acc_loss('./log/AlexnetReLU_log.txt')

    # train_acc3, val_acc3, train_loss3, val_loss3, epochs = get_acc_loss('./log/AlexnetReLUSwish_log.txt')
    train_acc4, val_acc4, train_loss4, val_loss4, epochs = get_acc_loss('./log/AlexnetSwish_log.txt')

    train_acc5, val_acc5, train_loss5, val_loss5, epochs = get_acc_loss('./log/AlexnetReLUSwish1_log.txt')

    # 1.比较TrainingAcc
    plot_dict = {
                 'MishTrainAcc': [train_acc1[250:], 'r'],
                 'ReLUTrainAcc': [train_acc2[250:], 'b'],
                 # 'ReLUSwishTrainAcc': [train_acc3, 'g'],
                 'SwishTrainAcc': [train_acc4[250:], 'y'],
                 'ReLUSwish1TrainAcc': [train_acc5[250:], 'c']
                }
    mean_train_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_train_acc = sorted(mean_train_acc, key=lambda x: x[1])
    print('AlexNetCompareResult:')
    print(mean_train_acc)
    compare_data(plot_dict, epochs[250:])

    # 2.比较ValAcc
    plot_dict = {'MishValAcc': [val_acc1[250:], 'r'],
                 'ReLUValAcc': [val_acc2[250:], 'b'],
                 # 'ReLUSwishValAcc': [val_acc3, 'g'],
                 'SwishValAcc': [val_acc4[250:], 'y'],
                 'ReLUSwish1ValAcc': [val_acc5[250:], 'c']
                 }
    mean_val_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_val_acc = sorted(mean_val_acc, key=lambda x: x[1])
    print(mean_val_acc)
    compare_data(plot_dict, epochs[250:])

    # 3.比较TrainLoss
    plot_dict = {'MishTrainLoss': [train_loss1[250:], 'r'],
                 'ReLUTrainLoss': [train_loss2[250:], 'b'],
                 # 'ReLUSwishValAcc': [val_acc3, 'g'],
                 'SwishTrainLoss': [train_loss4[250:], 'y'],
                 'ReLUSwish1TrainLoss': [train_loss5[250:], 'c']
                 }
    mean_train_loss = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_train_loss = sorted(mean_train_loss, key=lambda x: x[1])
    print(mean_train_loss)
    compare_data(plot_dict, epochs[250:])

    # 4.比较ValLoss
    plot_dict = {'MishValLoss': [val_loss1[250:], 'r'],
                 'ReLUValLoss': [val_loss2[250:], 'b'],
                 # 'ReLUSwishValAcc': [val_acc3, 'g'],
                 'SwishValLoss': [val_loss4[250:], 'y'],
                 'ReLUSwish1ValLoss': [val_loss5[250:], 'c']
                 }
    mean_val_loss = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_val_loss = sorted(mean_val_loss, key=lambda x: x[1])
    print(mean_val_loss)
    compare_data(plot_dict, epochs[250:])


def InceptionV2Compare():
    train_acc1, val_acc1, train_loss1, val_loss1, epochs = get_acc_loss('./log/InceptionV2Mish_log.txt')
    train_acc2, val_acc2, train_loss2, val_loss2, epochs = get_acc_loss('./log/InceptionV2ReLU_log.txt')

    train_acc3, val_acc3, train_loss3, val_loss3, epochs = get_acc_loss('./log/InceptionV2ReLUSwish1_log.txt')
    train_acc4, val_acc4, train_loss4, val_loss4, epochs = get_acc_loss('./log/InceptionV2Swish_log.txt')

    # 1.比较TrainingAcc
    plot_dict = {'MishTrainAcc': [train_acc1, 'r'], 'ReLUTrainAcc': [train_acc2, 'b'],
                 'ReLUSwishTrainAcc': [train_acc3, 'g'], 'SwishTrainAcc': [train_acc4, 'y']}
    mean_train_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_train_acc = sorted(mean_train_acc, key=lambda x: x[1])
    print('InceptionV2CompareResult:')
    print(mean_train_acc)
    compare_data(plot_dict, epochs)

    # 2.比较ValAcc
    plot_dict = {'MishValAcc': [val_acc1, 'r'], 'ReLUValAcc': [val_acc2, 'b'],
                 'ReLUSwishValAcc': [val_acc3, 'g'], 'SwishValAcc': [val_acc4, 'y']}
    mean_val_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_val_acc = sorted(mean_val_acc, key=lambda x: x[1])
    print(mean_val_acc)
    compare_data(plot_dict, epochs)


def ReLUSwishCompare():
    train_acc1, val_acc1, train_loss1, val_loss1, epochs = get_acc_loss('./log/InceptionV2ReLUSwish_log.txt')
    train_acc2, val_acc2, train_loss2, val_loss2, epochs = get_acc_loss('./log/InceptionV2ReLUSwish1_log.txt')

    # 1.比较TrainingAcc
    plot_dict = {'ReLUSwishTrainAcc': [train_acc1, 'r'], 'ReLUSwish1TrainAcc': [train_acc2, 'b'],}
                 # 'ReLUSwishTrainAcc': [train_acc3, 'g'], 'SwishTrainAcc': [train_acc4, 'y']}
    mean_train_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_train_acc = sorted(mean_train_acc, key=lambda x: x[1])
    print('ReLUSwishCompareResult:')
    print(mean_train_acc)
    compare_data(plot_dict, epochs)

    # 2.比较ValAcc
    plot_dict = {'ReLUSwishValAcc': [val_acc1, 'r'], 'ReLUSwish1ValAcc': [val_acc2, 'b'],}
                 # 'ReLUSwishValAcc': [val_acc3, 'g'], 'SwishValAcc': [val_acc4, 'y']}
    mean_val_acc = [(k, np.mean(v[0])) for k, v in plot_dict.items()]
    mean_val_acc = sorted(mean_val_acc, key=lambda x: x[1])
    print(mean_val_acc)
    compare_data(plot_dict, epochs)


def ActivationCompare():
    """
    激活函数比较
    :return:
    """
    inputs = tf.constant(np.linspace(-4, 4, 101))

    # 1. ReLU
    relu = K.relu(inputs)

    # 2. ReLUSwish
    condition = K.greater(inputs, 0)
    relu_swish = tf.where(condition, inputs, 2 * inputs * K.sigmoid(inputs))

    # 3. Swish
    # swish = inputs * K.sigmoid(inputs)
    swish = tf.nn.swish(inputs)

    # 4. Mish
    mish = inputs * K.tanh(K.softplus(inputs))

    plot_dict = {'ReLU': [relu.numpy(), 'r'], 'ReLUSwish': [relu_swish.numpy(), 'b'],
                 'Swish': [swish.numpy(), 'g'], 'Mish': [mish.numpy(), 'y']}
    compare_data(plot_dict, inputs.numpy(), save_path='acitvations_pic/activation1.svg')


def Activation2Compare():
    """
    激活函数比较
    :return:
    """
    inputs = tf.constant(np.linspace(-10, 10, 101))

    # 1. ReLU
    relu = K.relu(inputs)

    # 2. LeakyReLU
    leaky_relu = tf.nn.leaky_relu(inputs, alpha=0.01)

    # 3. ReLU6
    relu6 = tf.nn.relu6(inputs)

    # 4. Mish
    mish = inputs * K.tanh(K.softplus(inputs))

    plot_dict = {'ReLU': [relu.numpy(), 'r'], 'LeakyReLU': [leaky_relu.numpy(), 'b'],
                 'ReLU6': [relu6.numpy(), 'g'], 'Mish': [mish.numpy(), 'y']}
    compare_data(plot_dict, inputs.numpy(), save_path='acitvations_pic/activation_relus.svg')


if __name__ == r'__main__':
    # get_acc_loss('./log/xception_log.txt')
    # XceptionCompare()
    AlexNetCompare()
    # ActivationCompare()
    # InceptionV2Compare()
    # ReLUSwishCompare()
    # Activation2Compare()
