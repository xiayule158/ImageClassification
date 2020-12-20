"""
日期修改
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

root_dir = '/media/xiayule/bdcp/other'


def modify_date():
    img_path = os.path.join(root_dir, '3.jpg')
    img = cv2.imread(img_path)
    # _, img1 = cv2.threshold(img, 150, 200, cv2.THRESH_BINARY)
    hue_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_range = np.array([140, 43, 46])
    h_range = np.array([180, 255, 255])
    th = cv2.inRange(hue_img, l_range, h_range)
    index1 = th == 255
    img1 = np.zeros(img.shape, np.uint8)
    img1[:, :] = (255, 255, 255)
    img1[index1] = img[index1]

    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.imshow('1', img1)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_print():
    """

    :return:
    """
    img_path = os.path.join(root_dir, 'zhuangbei2.jpg')
    img = cv2.imread(img_path)

    w, h, c = img.shape
    dst_img = np.ones((w, h, c), np.uint8)*255
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2BGRA)

    for i in range(w):
        for j in range(h):
            pixel = img[i, j, :]
            b, g, r = pixel[0], pixel[1], pixel[2]
            if 80 <= b < 160 and 80 <= g < 150 and 140 <= r < 240:
                dst_img[i, j, 0] = b
                dst_img[i, j, 1] = g
                dst_img[i, j, 2] = r
                # dst_img[i, j, 3] = [b, g, r]
            else:
                dst_img[i, j, 3] = 0

    cv2.imwrite(os.path.join(root_dir, 'zhuangbei2.png'), dst_img)
    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.imshow('1', dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_print1():
    """

    :return:
    """
    img_path = os.path.join(root_dir, 'zhuangbei2.jpg')
    img = cv2.imread(img_path)

    w, h, c = img.shape
    dst_img = np.ones((w, h, c), np.uint8)*255
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2BGRA)

    for i in range(w):
        for j in range(h):
            pixel = img[i, j, :]
            b, g, r = pixel[0], pixel[1], pixel[2]
            m = (int(b)+int(g)+int(r))/3
            if abs(b-m) < 20 and abs(g-m) < 20 and abs(r-m) < 20:
                dst_img[i, j, 3] = 0
            else:
                dst_img[i, j, 0] = b
                dst_img[i, j, 1] = g
                dst_img[i, j, 2] = r

    cv2.imwrite(os.path.join(root_dir, 'zhuangbei2.png'), dst_img)
    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.imshow('1', dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_touming():
    """

    :return:
    """
    img_path = os.path.join(root_dir, '26.jpg')
    img = cv2.imread(img_path)

    w, h, c = img.shape
    dst_img = np.ones((w, h, c), np.uint8)*255
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2BGRA)

    for i in range(w):
        for j in range(h):
            pixel = img[i, j, :]
            b, g, r = pixel[0], pixel[1], pixel[2]
            if 0 <= b < 50 and 0 <= g < 50 and 0 <= r < 50:
                dst_img[i, j, 0] = b
                dst_img[i, j, 1] = g
                dst_img[i, j, 2] = r
                # dst_img[i, j, 3] = [b, g, r]
            else:
                dst_img[i, j, 3] = 0

    cv2.imwrite(os.path.join(root_dir, '26_1.png'), dst_img)
    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.imshow('1', dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def myfunc1(x):
    if x >= 0:
        return x
    else:
        return 2*x/(1+np.exp(-x))


def myfunc1_der1(x):
    if x >= 0:
        return 1
    else:
        return 2*(1 + np.exp(-x) + x * np.exp(-x)) / pow(1 + np.exp(-x), 2)


def plot_swish():
    """
    swish图像
    :return:
    """
    x = np.linspace(-4, 4, 1001)
    y = np.array([myfunc1(i) for i in x])
    y_d1 = np.array([myfunc1_der1(i) for i in x])
    plt.plot(x, y, x, y_d1)
    plt.show()


def modify_pixel():
    img_path = os.path.join(root_dir, '51.png')
    img = cv2.imread(img_path).astype('int')

    w, h, c = img.shape
    dst_img = np.ones((w, h, c), np.uint8) * 255
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2BGRA)

    for i in range(w):
        for j in range(h):
            pixel = img[i, j, :]
            b, g, r = pixel[0], pixel[1], pixel[2]

            if b < 255 and g < 255 and r < 255:
                dst_img[i, j, 0] = b
                dst_img[i, j, 1] = g
                dst_img[i, j, 2] = r+15
                # dst_img[i, j, 3] = [b, g, r]
            else:
                dst_img[i, j, 3] = 0
    dst_img[dst_img > 255] = 255
    cv2.imwrite(os.path.join(root_dir, '5_1.png'), dst_img)
    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.imshow('1', dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == r'__main__':
    get_touming()
    # plot_swish()

    # modify_pixel()