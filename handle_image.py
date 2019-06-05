import cv2
import os

image_path = 'C:\\Users\\15479\\Desktop\\CASIA-FaceV5'
face_model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


def make_path():
    # 创建图片保存路径
    # 当原路径有图片时删除
    paths = ['train_image2', 'test_image2']
    for path in paths:
        if os.path.exists(path):
            ls = os.listdir(path)
            for i in ls:
                c_path = os.path.join(path, i)
                os.remove(c_path)
        else:
            os.makedirs(path)


def save_face(img, name, x, y, width, height):
    # 保存获取到的人脸,将图片统一转换为100*100
    image = img[y:y + height, x:x + width]
    resized_image = cv2.resize(image, (100, 100))
    cv2.imwrite(name, resized_image)


def handle_image():
    # 处理原始图片

    # 准备图片保存路径
    make_path()

    # 具体处理每个图片
    for i in range(500):
        image_path2 = image_path + '\\' + str(i).zfill(3)
        p_image = []
        for j in range(5):
            image_path3 = image_path2 + '\\' + str(i).zfill(3) + '_' + str(j) + '.bmp'
            image = cv2.imread(image_path3)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_model.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3,
                                                minSize=(5, 5))
            # 储存人脸图片，按照4：1的比例，4份放入训练集，1份放入测试集
            if j == 0:
                face_path = 'test_image2'
            else:
                face_path = 'train_image2'
            for (x, y, width, height) in faces:
                p_image.append([image, '%s\\%d_%d.bmp' % (face_path, i, j), x, y, width, height])
        if len(p_image) == 5:
            for a_image, name, x, y, width, height in p_image:
                save_face(a_image, name, x, y, width, height)


if __name__ == '__main__':
    handle_image()
