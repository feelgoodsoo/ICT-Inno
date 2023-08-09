from keras.layers import Dense, Flatten
from keras import Sequential
from keras.preprocessing import image

dataGen = image.ImageDataGenerator(rescale=1./255)
trainGen = dataGen.flow_from_directory(
    './image/cat/cat1.jpg', target_size=(200, 200))
print(trainGen.class_indices)

model = Sequential()
model.add(Flatten(input_shape=(200, 200, 3)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
h = model.fit(trainGen, epochs=50)
model.save('catdog.h5')
