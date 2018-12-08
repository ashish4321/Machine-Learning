import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# Importing the Keras libraries and packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(labels = ["label"],axis = 1) 
Y_train = train["label"]

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)

# Initialising the CNN
classifier = Sequential()

# 1st Convolution Layer
classifier.add(Convolution2D(32, (3, 3), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 2nd Convolutional Layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(activation = 'relu', units=256))
classifier.add(Dropout(0.4))
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dropout(0.4))
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dropout(0.4))

classifier.add(Dense(activation = 'relu', units=10))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(X_train)

classifier.fit_generator(datagen.flow(X_train,Y_train),
                         nb_epoch = 20,
                         validation_data = (X_test,Y_test))
