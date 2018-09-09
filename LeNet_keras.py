from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def get_keras_model():
    model = Sequential()

    #1st Layer - Add a flatten layer
    model.add(Conv2D(32, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    #2nd Layer - Add a fully connected layer
    model.add(Dense(128))
    #3rd Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    #4th Layer - Add a fully connected layer
    model.add(Dense(5))

    #5th Layer - Add a ReLU activation layer
    model.add(Activation('softmax'))
    return model