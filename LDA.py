import numpy as np
from base_func import *


def lda(train_images, train_labels, k):
    # 利用lda算法进行降维

    labels = list(set(train_labels))

    image_classify = {}
    for label in labels:
        X1 = np.array([train_images[i] for i in range(len(train_images)) if train_labels[i] == label])
        image_classify[label] = X1

    mju = np.mean(train_images, axis=0)
    mju_classify = {}

    # 各类的平均值
    for label in labels:
        mju1 = np.mean(image_classify[label], axis=0)
        mju_classify[label] = mju1

    # St = np.dot((X - mju).T, X - mju)
    # 计算类内散度矩阵
    # 各类的图片数据减去其平均值
    Sw = np.zeros((len(mju), len(mju)))
    for i in labels:
        data = image_classify[i] - mju_classify[i]
        Sw += np.dot(data.T, data)

    # Sb=St-Sw
    # 计算类内散度矩阵
    # 各类平均值减总体平均值
    Sb = np.zeros((len(mju), len(mju)))
    for i in labels:
        data = (mju_classify[i] - mju).reshape((len(mju), 1))
        Sb += len(image_classify[i]) * np.dot(data, data.T)

    # 计算Sw-1*Sb的特征值和特征矩阵
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    sorted_indices = np.argsort(eig_vals)
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]

    return topk_eig_vecs


def get_lda_model():
    train_images, train_labels = get_image_data("train")

    topk_eig_vecs = lda(train_images, train_labels, k=1910)
    save_model(topk_eig_vecs, "lda_model_1910")


def get_lda_result(al=None, k=3):
    train_images, train_labels = get_image_data("train")
    test_images, test_labels = get_image_data("test")

    topk_eig_vecs = np.array(read_model2("lda_model_1910"))
    topk_eig_vecs = np.array(topk_eig_vecs, dtype=np.complex)

    num = 1820 if model_path == "model2" else 940
    train_images = np.dot(train_images, topk_eig_vecs[:, :num])
    test_images = np.dot(test_images, topk_eig_vecs[:, :num])

    # 使用欧式距离或其他方式识别图片
    if not al:
        predict_labels = euclidean_distance(train_images, train_labels, test_images)
    else:
        predict_labels = al(train_images, train_labels, test_images, k)

    accuracy = np.sum(predict_labels == np.array(test_labels))/len(test_labels)
    accuracy = "%.2f%%" % (accuracy*100)

    # 返回训练图片数、测试图片数、正确率
    return len(train_labels), len(test_labels), accuracy


def get_one_lda_result(test_image_name, al=None):
    train_images, train_labels = get_image_data("train")
    test_image, test_label = get_test_image_data(test_image_name)

    topk_eig_vecs = np.array(read_model2("lda_model_1910"))
    topk_eig_vecs = np.array(topk_eig_vecs, dtype=np.complex)

    num = 1820 if model_path == "model2" else 940
    train_images = np.dot(train_images, topk_eig_vecs[:, :num])
    test_image = np.dot(test_image, topk_eig_vecs[:, :num])

    # 使用欧式距离或其他方式识别图片
    if not al:
        predict_label = euclidean_distance(train_images, train_labels, test_image)
    else:
        predict_label = al(train_images, train_labels, test_image)

    # 返回实际标签、预测标签
    return test_label[0], predict_label[0]


if __name__ == '__main__':
    get_lda_model()
    print(get_lda_result())
    print(get_one_lda_result("0_0.bmp"))
