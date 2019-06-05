import os
import cv2
import numpy as np

# # att_face数据集
# image_path = {'train': 'train_image', 'test': 'test_image'}
# model_path = "model"

# CASIA-FaceV5数据集
image_path = {'train': 'train_image2', 'test': 'test_image2'}
model_path = "model2"

# # CASIA-FaceV5数据集 取40个
# image_path = {'train': 'train_image3', 'test': 'test_image3'}
# model_path = "model3"

# # att_face数据集 取40*5个
# image_path = {'train': 'train_image4', 'test': 'test_image4'}
# model_path = "model4"


def get_image_data(path, flag=True):
    # 根据路径读取训练和测试图片信息

    images = []
    labels = []

    ls = os.listdir(image_path[path])
    for path2 in ls:
        c_path = os.path.join(image_path[path], path2)
        a_image = cv2.imread(c_path)
        a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2GRAY)
        # if model_path != "model":
        #     # 直方图均衡化
        #     a_image = cv2.equalizeHist(a_image)
        a_image = cv2.equalizeHist(a_image)
        if flag:
            a_image = np.reshape(a_image, (-1))
        images.append(a_image)
        labels.append(c_path.split("\\")[-1].split("_")[0])
    return images, labels


def get_image_data_cnn(path):
    # 根据路径读取训练和测试图片信息

    images = []
    labels = []

    ls = os.listdir(image_path[path])
    for i in ls:
        c_path = os.path.join(image_path[path], i)
        a_image = cv2.imread(c_path)
        a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2GRAY)
        # if model_path != "model":
        #     a_image = cv2.equalizeHist(a_image)
        a_image = cv2.equalizeHist(a_image)
        images.append(np.asarray(a_image, dtype=np.float32).reshape(len(a_image), len(a_image[0]), 1))
        labels.append(c_path.split("\\")[-1].split("_")[0])
    images = np.asarray(images) / 255.0
    from keras.utils import np_utils
    ont_hot_labels = np_utils.to_categorical(np.asarray(labels))
    return np.array(images), ont_hot_labels


def get_test_image_data(file_name, flag=True):
    # 根据路径读取测试图片信息
    c_path = os.path.join(image_path['test'], file_name)
    a_image = cv2.imread(c_path)
    a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2GRAY)
    # if model_path != "model":
    #     # 直方图均衡化
    #     a_image = cv2.equalizeHist(a_image)
    a_image = cv2.equalizeHist(a_image)
    if flag:
        a_image = [np.reshape(a_image, (-1))]
    else:
        a_image = [a_image]
    a_label = [file_name.split("_")[0]]
    return a_image, a_label


def read_model2(name):
    data = []
    with open("%s\\%s" % (model_path, name)) as s:
        for line in s:
            if line.strip():
                data.append([_[1:-1] for _ in line.strip().split()])
    return data


def read_model(name, data_type=np.float):
    data = []
    with open("%s\\%s" % (model_path, name)) as s:
        for line in s:
            if line.strip():
                data.append([_ for _ in line.strip().split()])
    return np.array(data, dtype=data_type)


def save_model(data, name):
    data = np.array(data)
    f = open("%s\\%s" % (model_path, name), 'w')
    for low in data:
        f.write(' '.join([str(_) for _ in low]) + '\n')
    f.close()


def euclidean_distance(train_images, train_labels, test_images):
    predict_labels = []
    for test_image in test_images:
        dis_list = []
        for train_image in train_images:
            dis_list.append(np.linalg.norm(test_image - train_image))
        sort_index = np.argsort(dis_list)
        predict_labels.append(train_labels[sort_index[0]])
    return predict_labels


