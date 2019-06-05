# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
from base_func import *


def pca(train_images, k):
    # 均值化矩阵
    mean_images = np.mean(np.mat(train_images), 0)
    train_images = train_images - np.tile(mean_images, (train_images.shape[0], 1))
    # 计算特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(train_images * train_images.T)
    # 取出指定个数的前n大的特征值
    eig_val_ind = eig_vects[:, :k]  # 取前k个特征向量
    eig_val_ind = train_images.T * eig_val_ind
    for i in range(k):  # 特征向量归一化
        eig_val_ind[:, i] /= np.linalg.norm(eig_val_ind[:, i])
    return np.array(train_images * eig_val_ind), mean_images, eig_val_ind


def get_pca_model():
    # 获取当前训练好的pca模型
    train_images, train_labels = get_image_data("train")
    low_mat, mean_images, eig_val_ind = pca(np.array(train_images), 145)
    save_model(mean_images, 'pca_mean_images')
    save_model(eig_val_ind, 'pca_eig_val_ind')


def get_pca_result(al=None, k=3):
    # 加载图片数据
    train_images, train_labels = get_image_data("train")
    test_images, test_labels = get_image_data("test")

    # 加载已训练好的模型数据
    mean_images = read_model("pca_mean_images")
    eig_val_ind = read_model("pca_eig_val_ind")

    # 对数据进行一定转化
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_images = np.dot(train_images - np.tile(mean_images, (train_images.shape[0], 1)), eig_val_ind)
    test_images = np.dot(test_images - np.tile(mean_images, (test_images.shape[0], 1)), eig_val_ind)

    # 使用欧式距离或其他方式识别图片
    if not al:
        predict_labels = euclidean_distance(train_images, train_labels, test_images)
    else:
        predict_labels = al(train_images, train_labels, test_images, k)

    accuracy = np.sum(predict_labels == np.array(test_labels))/len(test_labels)
    accuracy = "%.2f%%" % (accuracy*100)

    # 返回训练图片数、测试图片数、正确率
    return len(train_labels), len(test_labels), accuracy


def get_one_pca_result(test_image_name, al=None):
    # 加载图片数据
    train_images, train_labels = get_image_data("train")
    test_image, test_label = get_test_image_data(test_image_name)

    # 加载已训练好的模型数据
    mean_images = read_model("pca_mean_images")
    eig_val_ind = read_model("pca_eig_val_ind")

    # 对数据进行一定转化
    train_images = np.array(train_images)
    test_image = np.array(test_image)
    train_images = np.dot(train_images - np.tile(mean_images, (train_images.shape[0], 1)), eig_val_ind)
    test_image = np.dot(test_image - np.tile(mean_images, (test_image.shape[0], 1)), eig_val_ind)

    # 使用欧式距离或其他方式识别图片
    if not al:
        predict_label = euclidean_distance(train_images, train_labels, test_image)
    else:
        predict_label = al(train_images, train_labels, test_image)

    # 返回实际标签、预测标签
    return test_label[0], predict_label[0]


if __name__ == '__main__':
    get_pca_model()
    print(get_pca_result())
    print(get_one_pca_result("0_0.bmp"))
