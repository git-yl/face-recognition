# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import math
import copy
from base_func import *


g_mapping = [
    0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
    11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
    16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
    22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
    29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
    36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
    42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
    47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57]


def min_binary(pixel):
    # 获得二进制特征的最小值
    length = len(pixel)
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '0'


def lbp(image, radius=2, count=8):
    # 得到图像的LBP特征
    # 计算某点边缘的像素位置
    dh = np.round([radius * math.sin(i * 2 * math.pi / count) for i in range(count)])
    dw = np.round([radius * math.cos(i * 2 * math.pi / count) for i in range(count)])

    height, width = image.shape
    image_lbp = np.zeros(image.shape, dtype=np.int)
    for x in range(radius, width - radius):
        for y in range(radius, height - radius):
            repixel = ''
            pixel = int(image[y, x])
            for h, w in zip(dh, dw):
                if int(image[int(y+h), int(x+w)]) > pixel:
                    repixel += '1'
                else:
                    repixel += '0'
            image_lbp[y, x] = int(min_binary(repixel), base=2)
    return image_lbp


def cal_lbp_histogram(image_lbp, hCount=5, wCount=5, maxLbpValue=255):
    # 分块计算lbp直方图
    height, width = image_lbp.shape
    res = np.zeros((hCount * wCount, max(g_mapping) + 1))

    for h in range(hCount):
        for w in range(wCount):
            blk = image_lbp[int(height * h / hCount):int(height * (h + 1) / hCount),
                            int(width * w / wCount):int(width * (w + 1) / wCount)]
            hist1 = np.bincount(blk.ravel(), minlength=maxLbpValue)

            # 进一步减少直方图维度为59维
            hist = res[h * wCount + w, :]
            for v, k in zip(hist1, g_mapping):
                hist[k] += v
            # hist /= hist.sum()
    return res


def get_dis(train_image, test_image):
    value = 0
    test_1 = copy.deepcopy(test_image)
    train_1 = copy.deepcopy(train_image)
    for i in range(math.ceil(math.log(len(test_1), 2))):
        value += np.sum([min(test, train) for test, train in zip(test_1, train_1)]) / (2**i)
        test_1 = [(test_1[2*j] + (test_1[2*j+1] if (2*j+1) < len(test_1) else 0)) for j in range((len(test_1)+1)//2)]
        train_1 = [(train_1[2*j] + (train_1[2*j+1] if (2*j+1) < len(train_1) else 0)) for j in range((len(train_1)+1)//2)]
        print(i, value)
    return value


def histogram_intersection_kernel(train_images, train_labels, test_images, test_labels):
    # 利用直方图交叉核计算相似度
    right_num = 0
    for test_image, test_label in zip(test_images, test_labels):
        print(test_label)
        dis_list = []
        for train_image in train_images:
            dis_list.append(get_dis(train_image, test_image))
        sort_index = np.argsort(dis_list)
        if test_label == train_labels[sort_index[0]]:
            right_num += 1
    print(right_num / len(test_labels))


def get_lbp_model():
    train_images, train_labels = get_image_data("train", False)
    test_images, test_labels = get_image_data("test", False)

    train_images = np.array([cal_lbp_histogram(lbp(image)).ravel() for image in train_images])
    test_images = np.array([cal_lbp_histogram(lbp(image)).ravel() for image in test_images])

    save_model(train_images, "lbp_train")
    save_model(test_images, "lbp_test")


def get_lbp_result(al=None, k=3):
    # 加载图片数据
    train_images, train_labels = get_image_data("train")
    test_images, test_labels = get_image_data("test")

    # 加载已训练好的模型数据
    train_images = read_model("lbp_train")
    test_images = read_model("lbp_test")

    # 使用欧式距离或其他方式识别图片
    if not al:
        predict_labels = euclidean_distance(train_images, train_labels, test_images)
    else:
        predict_labels = al(train_images, train_labels, test_images, k)

    accuracy = np.sum(predict_labels == np.array(test_labels)) / len(test_labels)
    accuracy = "%.2f%%" % (accuracy * 100)

    # histogram_intersection_kernel(train_images, train_labels, test_images, test_labels)

    # 返回训练图片数、测试图片数、正确率
    return len(train_labels), len(test_labels), accuracy


def get_one_lbp_result(test_image_name, al=None):
    # 加载图片数据
    train_images, train_labels = get_image_data("train", False)
    test_image, test_label = get_test_image_data(test_image_name, False)

    # 加载已训练好的模型数据
    train_images = read_model("lbp_train")

    test_image = np.array([cal_lbp_histogram(lbp(test_image[0])).ravel()])

    # 使用欧式距离或其他方式识别图片
    if not al:
        predict_label = euclidean_distance(train_images, train_labels, test_image)
    else:
        predict_label = al(train_images, train_labels, test_image)

    # 返回实际标签、预测标签
    return test_label[0], predict_label[0]


if __name__ == '__main__':
    get_lbp_model()
    print(get_lbp_result())
    print(get_one_lbp_result("0_0.bmp"))
