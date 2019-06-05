import numpy as np
import os
import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from base_func import *


def cnn(train_images, train_labels, test_images, test_labels):
    # 构建cnn模型
    face_model = keras.Sequential()
    # 添加卷积层
    face_model.add(Conv2D(64, kernel_size=(3, 3),
                          input_shape=(train_images.shape[1], train_images.shape[2], 1), activation='relu'))
    # 添加池化层
    face_model.add(MaxPooling2D(pool_size=(3, 3)))
    # 添加Dropout层
    face_model.add(Dropout(0.2))
    # 添加Flatten层
    face_model.add(Flatten())
    # 添加全连接层
    face_model.add(Dense(512, activation='relu'))

    face_model.add(Dropout(0.4))
    # 添加输出层
    face_model.add(Dense(len(train_labels[0]), activation='softmax'))
    # 打印模型结构
    face_model.summary()

    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.8
    nesterov = True
    sgd_optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
    face_model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])

    batch_size = 102 if model_path == 'model2' else 40
    epochs = 200 if model_path == 'model2' else (30 if model_path == 'model' else 50)
    for i in range(1):
        face_model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                       shuffle=True, validation_data=(test_images, test_labels))
        face_model.evaluate(test_images, test_labels, verbose=0)

    model_path2 = model_path + '\\cnn_model.h5'
    face_model.save(model_path2)


def get_cnn_model():
    train_images, train_labels = get_image_data_cnn("train")
    test_images, test_labels = get_image_data_cnn("test")

    cnn(train_images, train_labels, test_images, test_labels)


def get_cnn_result():
    # 加载图片数据
    train_images, train_labels = get_image_data_cnn("train")
    test_images, test_labels = get_image_data_cnn("test")

    keras.backend.clear_session()
    model_path2 = model_path + '\\cnn_model.h5'
    from keras.models import load_model
    face_model = load_model(model_path2)

    test_result = face_model.evaluate(test_images, test_labels, verbose=0)

    accuracy = "%.2f%%" % (test_result[1]*100)

    # 返回训练图片数、测试图片数、正确率
    return len(train_labels), len(test_labels), accuracy


def get_one_cnn_result(test_image_name):
    # 加载图片数据
    test_image, test_label = get_test_image_data(test_image_name, False)
    test_image = np.array(test_image)
    test_image = np.array(np.asarray(test_image) / 255.0).reshape(1, test_image.shape[1], test_image.shape[2], 1)

    # 加载模型
    keras.backend.clear_session()
    model_path2 = model_path + '\\cnn_model.h5'
    from keras.models import load_model
    face_model = load_model(model_path2)

    # 获取预测值
    test_result = face_model.predict(test_image, verbose=0)
    index = np.argmax(test_result)

    # 返回实际标签、预测标签
    return test_label[0], str(index)


if __name__ == '__main__':
    get_cnn_model()
    print(get_cnn_result())
    print(get_one_cnn_result("15_0.bmp"))
