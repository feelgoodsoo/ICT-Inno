from flask import Flask, request, render_template, redirect
import numpy as np
from flasgger import Swagger
import pickle as pkl
from keras.models import load_model

app = Flask(__name__)
swagger = Swagger(app)

model = load_model('./models/iris.h5')


@app.route('/predict', methods=['POST'])
def predict():
    """ Endpoint taking one input
    ---
    parameters:
        - name: Sepal Length
          in: query
          type: number
          required: true
        - name: Sepal Width
          in: query
          type: number
          required: true
        - name: Petal Length
          in: query
          type: number
          required: true
        - name: Petal Width
          in: query
          type: number
          required: true
    responses:
        200:
            description: "0: Setosa, 1: Versicolour, 2: Virginica"
    """

    s_length = float(request.args.get("Sepal Length"))
    s_width = float(request.args.get("Sepal Width"))
    p_length = float(request.args.get("Petal Length"))
    p_width = float(request.args.get("Petal Width"))

    pred = model.predict([[s_length, s_width, p_length, p_width]])
    # print(pred)
    p = int(pred.argmax(axis=1))
    print('분류예측: ', p)
    print('확률값: ', pred.max(axis=1))

    return str(p)


@app.route('/irisForm')
def irisForm():
    return render_template('irisForm.html')


@app.route('/piris')
def piris():
    s_length = request.args.get("sl")  # request.args["Sepal Length"]
    s_width = request.args.get("sw")
    p_length = request.args.get("pl")
    p_width = request.args.get("pw")
    print(s_length, s_width)
    input_features = np.array(
        [[float(s_length), float(s_width), float(p_length), float(p_width)]])
    pred = model.predict(input_features)
    print('분류예측', pred.argmax(axis=1))
    print('확률값', pred.max(axis=1))
    p = int(pred.argmax(axis=1))  # [[0]]

    result = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
    return render_template('piris.html', result=result[p])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500, debug=True)
