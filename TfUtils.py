import os
from glob import glob
from IPython import display
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import random

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
image_root = './data'

height, width, channels = 224, 224, 3


def play_curve(epochs, train_acc_loss, val_acc_loss, fig_path='', acc=True):
    """
    保存图片文件路径
    :param epochs:
    :param train_acc_loss:
    :param val_acc_loss:
    :param fig_path:
    :param acc:
    :return:
    """
    display.set_matplotlib_formats('svg')  # 矢量图
    plt.rcParams['figure.figsize'] = (14.4, 7.91)  # 图片尺寸
    plt.xlabel('epoch')  # 横坐标

    if acc:
        plt.ylabel('acc')    # 纵坐标
        plt.title('train acc')
        plt.plot(epochs, train_acc_loss, label='train_acc')
        plt.plot(epochs, val_acc_loss, label='val_acc')
        # plt.legend(['train_acc', 'val_acc'])
        plt.legend()
    else:
        plt.ylabel('loss')  # 纵坐标
        plt.title('train loss')
        plt.plot(epochs, train_acc_loss, label='train_loss')
        plt.plot(epochs, val_acc_loss, label='val_loss')
        plt.legend(['train_loss', 'val_loss'])

    if fig_path:
        plt.savefig(fig_path)
    plt.show()


def get_data(batch_size=64):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        os.path.join(image_root, 'Training'),
        target_size=(height, width),
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        class_mode='categorical'
    )
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
    )
    val_generator = val_datagen.flow_from_directory(
        os.path.join(image_root, 'Valid'),
        target_size=(height, width),
        batch_size=batch_size,
        seed=7,
        shuffle=False,
        class_mode='categorical'
    )
    return train_generator, val_generator


def get_dataset(batch_size=64, img_shape=[224, 224, 3], epchos=100):
    """
    以dataset的形式生成训练数据和验证集数据
    :param batch_size:
    :param img_shape:
    :param epchos:
    :return:
    """
    # 0. define map function
    def load_process_image(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_shape[:2])
        img /= 255.0
        return img, label

    # 1.define some var
    labels = os.listdir(os.path.join(image_root, 'Training'))
    all_train_image_path, all_train_label = [],  []
    all_valid_image_path, all_valid_label = [], []

    # 2. get train data image list and label list
    for sub_path in glob(os.path.join(image_root, 'Training/*')):
        all_train_image_path[len(all_train_image_path):] = glob(os.path.join(sub_path, '*'))
    random.shuffle(all_train_image_path)
    all_train_label = [labels.index(img.split('/')[-2]) for img in all_train_image_path]
    all_train_label = tf.one_hot(all_train_label, len(labels))

    # 3. get valid data image list and label list
    for sub_path in glob(os.path.join(image_root, 'Valid/*')):
        all_valid_image_path[len(all_train_image_path):] = glob(os.path.join(sub_path, '*'))
    random.shuffle(all_valid_image_path)
    all_valid_label = [labels.index(img.split('/')[-2]) for img in all_valid_image_path]
    all_valid_label = tf.one_hot(all_valid_label, len(labels))

    # 4. generate train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((all_train_image_path, all_train_label))
    train_dataset = train_dataset.map(load_process_image)
    # train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.repeat(epchos).batch(batch_size)
    train_dataset = train_dataset.prefetch(batch_size)

    # 5. generate valid dataset
    valid_dataset = tf.data.Dataset.from_tensor_slices((all_valid_image_path, all_valid_label))
    valid_dataset = valid_dataset.map(load_process_image)
    # valid_dataset = valid_dataset.shuffle(10000)
    valid_dataset = valid_dataset.repeat(epchos).batch(batch_size)
    valid_dataset = valid_dataset.prefetch(batch_size)

    return train_dataset, valid_dataset, len(all_valid_image_path), len(all_valid_image_path)


if __name__ == r'__main__':
    # train_generator, val_generator = get_data()
    # print(train_generator.samples, val_generator.samples)
    # for i in range(1):
    #     x, y = val_generator.next()
    #     print(x.shape, y.shape)
    #     print(y)

    # train_gene, val_gene = get_data()
    train_set, valid_set = get_dataset()
