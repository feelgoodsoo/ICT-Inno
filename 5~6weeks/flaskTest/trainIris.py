from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
import pandas as pd


iris = load_iris()
# irisDF = pd.DataFrame( iris['data'])
# irisDF.columns = iris['feature_names']
# irisDF['species'] = iris['target']
# print( irisDF )
x_data = iris['data']
y_data = iris['target']
print(x_data)
dense = Dense(units=3, input_dim=4, activation="softmax")
model = Sequential([dense])
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.1),
              metrics=['acc'])
h = model.fit(x_data, y_data, epochs=500)
pred = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(pred)
print('분류예측: ', pred.argmax(axis=1))
print('확률값: ', pred.max(axis=1))

model.save('iris.h5')
