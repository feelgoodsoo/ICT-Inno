from keras.preprocessing import image
from keras.models import load_model

pilImg = image.load_img('./image/dog/dog2.jpg', target_size=(200, 200))
imgArr = image.img_to_array(pilImg)/255
print(imgArr)
model = load_model('catdog.h5')
# print( model.summary() )


myD = {0: '고양이', 1: '강아지'}


pred = model.predict(imgArr.reshape(1, 200, 200, 3))
p = int(pred.argmax(axis=1))
print(myD[p], pred.round(2))
