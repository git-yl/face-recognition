from PCA import get_one_pca_result, get_pca_result
from LBP import get_one_lbp_result, get_lbp_result
from LDA import get_one_lda_result, get_lda_result
from cnn import get_one_cnn_result, get_cnn_result
from knn import al_knn


def test_one_image(al_type, test_image_name):
    # 各算法用于测试的函数的汇总
    al_func = {"pca": get_one_pca_result,
               "lbp": get_one_lbp_result,
               "lda": get_one_lda_result,
               "cnn": get_one_cnn_result}

    al_type = al_type.split("_")

    # 调用函数获取预测结果
    if al_type[0] != "knn":
        test_label, predict_label = al_func[al_type[0]](test_image_name)
    else:
        test_label, predict_label = al_func[al_type[1]](test_image_name, al_knn)

    return test_label, predict_label


def test_all_image(al_type):
    # 各算法用于测试的函数的汇总
    al_func = {"pca": get_pca_result,
               "lbp": get_lbp_result,
               "lda": get_lda_result,
               "cnn": get_cnn_result}

    al_type = al_type.split("_")

    # 调用函数获取预测结果
    if al_type[0] != "knn":
        train_label_num, test_label_num, accuracy = al_func[al_type[0]]()
    else:
        train_label_num, test_label_num, accuracy = al_func[al_type[1]](al_knn)

    return train_label_num, test_label_num, accuracy
