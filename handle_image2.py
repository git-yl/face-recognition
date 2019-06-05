import cv2
import os


image_path = 'C:\\Users\\15479\\Desktop\\att_faces'


def make_path():
    # 创建图片保存路径
    # 当原路径有图片时删除
    paths = ['train_image4', 'test_image4']
    for path in paths:
        if os.path.exists(path):
            ls = os.listdir(path)
            for i in ls:
                c_path = os.path.join(path, i)
                os.remove(c_path)
        else:
            os.makedirs(path)


def handle_image():
    # 处理原始图片

    # 准备图片保存路径
    make_path()

    # 具体处理每个图片
    for i in range(40):
        image_path2 = image_path + '\\s' + str(i+1)
        for j in range(5):
            image_path3 = image_path2 + '\\' + str(j+1) + '.pgm'
            image = cv2.imread(image_path3, 0)
            # 储存人脸图片，按照9：1的比例，9份放入训练集，1份放入测试集
            if j == 0:
                face_path = 'test_image4'
            else:
                face_path = 'train_image4'
            cv2.imwrite('%s\\%d_%d.bmp' % (face_path, i, j), image)


if __name__ == '__main__':
    handle_image()
