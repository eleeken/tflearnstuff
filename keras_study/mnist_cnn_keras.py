# -*- coding: utf-8 -*-

from keras.models import Sequential
models = Sequential()

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
#conv1
models.add(Conv2D(32, (5,5), activation='relu', input_shape = (28,28,1)))
models.add(MaxPooling2D(pool_size=(2,2)))
#conv2
models.add(Conv2D(64, (5,5), activation='relu'))
models.add(MaxPooling2D(pool_size=(2,2)))
#dropout
models.add(Dropout(0.25))
#Flatten
models.add(Flatten())
#fc1
models.add(Dense(512, activation='relu'))
models.add(Dropout(0.5))
#fc2
models.add(Dense(10, activation='softmax'))

from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
models.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_val = mnist.validation.images
y_val = mnist.validation.labels

x_train = x_train.reshape(-1, 28, 28,1)
x_val = x_val.reshape(-1,28,28,1)

models.fit(x=x_train, y=y_train, batch_size = 100, epochs = 10, shuffle=True, validation_data=(x_val, y_val))
