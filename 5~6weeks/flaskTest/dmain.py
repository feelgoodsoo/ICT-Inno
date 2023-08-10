from flask import Flask, request, render_template, redirect
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

category = ['고양이', '개']
model = load_model('./models/catdog.h5')


@app.route('/')
def index():
    return "flask"


@app.route('/imgselect')
def imgselect():
    return render_template('imgselect.html')


@app.route('/fsend', methods=['POST'])
def fsend():
    file = request.files['image']
    print('파일명:', file.filename)
    sName = './static/downimage/'+file.filename
    file.save(sName)

    if not file:
        return "<h1>파일이 없습니다.</h1>"

    pImg = image.load_img(sName, target_size=(200, 200))
    imgArr = image.img_to_array(pImg) / 255
    pred = model.predict(imgArr.reshape(1, 200, 200, 3))
    p = int(pred.argmax(axis=1))

    return render_template('predict.html', label=category[p])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500, debug=True)
