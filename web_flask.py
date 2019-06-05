from flask import Flask
from flask import request
from flask import Response
from flask import render_template
from test_image import test_all_image, test_one_image
from base_func import image_path

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('tab.html')


@app.route('/test-one')
def test_one():
    test_image_name = request.args.get('test_image')
    al_type = request.args.get('type')
    test_label, predict_label = test_one_image(al_type, test_image_name)
    return test_label + "|" + predict_label


@app.route('/test-all')
def test_all():
    al_type = request.args.get('type')
    train_label_num, test_label_num, accuracy = test_all_image(al_type)
    return "%d|%d|%s" % (train_label_num, test_label_num, accuracy)


@app.route('/image/<img_local_path>')
def return_img_stream(img_local_path):
    image = ''
    with open(image_path["train"] + "\\" + img_local_path + ".bmp", 'rb') as img_f:
        image = img_f.read()
    return Response(image)


if __name__ == '__main__':
    app.run()
