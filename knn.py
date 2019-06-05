from base_func import *
from collections import Counter
from PCA import get_pca_result, get_one_pca_result
from LBP import get_one_lbp_result, get_lbp_result
from LDA import get_one_lda_result, get_lda_result


def al_knn(train_images, train_labels, test_images, k=3):
    # 利用knn算法判断人脸的归属
    predicts = []
    for image in test_images:
        distances = []
        n_image_type = []
        for train_image in train_images:
            distances.append(np.linalg.norm(image - train_image))
        a = np.array(distances)
        b = np.argsort(a)
        for order in b[:k]:
            n_image_type.append(train_labels[order])
        if Counter(n_image_type).most_common(1)[0][1] > 1:
            predicts.append(Counter(n_image_type).most_common(1)[0][0])
        else:
            predicts.append(train_labels[b[0]])
    return predicts


def al_knn2(train_images, train_labels, test_images, k=3):
    # 利用knn算法判断人脸的归属
    predicts = []
    for image in test_images:
        distances = []
        n_image_type = []
        for train_image in train_images:
            distances.append(np.sum([(i-j)**2/abs(i+j) for i, j in zip(image, train_image)]))
        a = np.array(distances)
        b = np.argsort(a)
        for order in b[-k:]:
            n_image_type.append(train_labels[order])
        if Counter(n_image_type).most_common(1)[0][1] > 1:
            predicts.append(Counter(n_image_type).most_common(1)[0][0])
        else:
            predicts.append(train_labels[b[0]])
    return predicts


def get_knn_result():
    # 加载图片数据
    train_images, train_labels = get_image_data("train")
    test_images, test_labels = get_image_data("test")

    for k in range(1, 21):
        predict_labels = al_knn(train_images, train_labels, test_images, k)

        accuracy = np.sum(predict_labels == np.array(test_labels))/len(test_labels)
        accuracy = "%.2f%%" % (accuracy*100)

        print(k, accuracy)

    # 返回训练图片数、测试图片数、正确率
    return len(train_labels), len(test_labels), accuracy


def get_one_knn_result(test_image_name):
    # 加载图片数据
    train_images, train_labels = get_image_data("train")
    test_image, test_label = get_test_image_data(test_image_name)

    predict_label = al_knn(train_images, train_labels, test_image)

    # 返回实际标签、预测标签
    return test_label[0], predict_label[0]


def get_al_knn_results(method):
    for k in range(1, 21):
        print(k, method(al_knn, k))


if __name__ == '__main__':
    # get_al_knn_results(get_pca_result)
    get_al_knn_results(get_lbp_result)
    # get_al_knn_results(get_lda_result)
    # print(get_knn_result())
    # print(get_one_knn_result("1_0.bmp"))
    # print(get_pca_result(al_knn))
    # print(get_one_pca_result("1_0.bmp", al_knn))
    # print(get_lbp_result(al_knn))
    # print(get_one_lbp_result("1_0.bmp", al_knn))
    # print(get_lda_result(al_knn))
    # print(get_one_lda_result("1_0.bmp", al_knn))
