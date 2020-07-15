# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import Input,Dense,Dropout
from keras.layers import Conv2D,MaxPool2D,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors

# Data loading and preprocessing
from keras.datasets import mnist

def do_cnn_2d(X, Y, testX, testY ):
    # Building convolutional network
    inputs = Input(shape=( 28, 28, 1), name='inputs')
    network = Conv2D(32, (3,3), activation='relu')(inputs)
    network = MaxPool2D(pool_size=(2,2))(network)
    network = BatchNormalization()(network)
    network = Conv2D(64, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(2,2))(network)
    network = Flatten()(network)
    network = BatchNormalization()(network)
    network = Dense(128, activation='tanh')(network)
    network = Dropout(0.8)(network)
    network = Dense(256, activation='tanh')(network)
    network = Dropout(0.8)(network)
    predictions = Dense(10, activation='softmax')(network)

    # Training
    adam = optimizers.adam(lr=0.01)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy',
                          metrics=['accuracy'])
    model = model.fit(X,Y,validation_data=(testX,testY))

def do_dnn_1d(x_train, y_train,x_test , y_test):
    print ("DNN and 1d")

    # Building deep neural network
    input_layer = Input(shape=(784,))
    dense1 = Dense(64, activation='tanh')(input_layer)
    dropout1 = Dropout(0.8)(dense1)
    dense2 = Dense(64, activation='tanh')(dropout1)
    dropout2 = Dropout(0.8)(dense2)
    predictions = Dense(10, activation='softmax')(dropout2)

    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = optimizers.SGD(lr=0.1,decay=0.96)

    # Training
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.fit(X, Y, epochs=10, validation_data=(testX, testY))

def do_svm_1d(x_train, y_train,x_test, y_test):
    print ("SVM and 1d")
    clf = svm.SVC(decision_function_shape='ovo')
    print (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    #print metrics.confusion_matrix(y_test, y_pred)

def do_knn_1d(x_train, y_train,x_test, y_test):
    print ("KNN and 1d")
    clf = neighbors.KNeighborsClassifier(n_neighbors=15)
    print (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    #print metrics.confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    print ("Hello MNIST")

    # do_dnn_1d
    # print(testX)
    # (X, Y), (testX, testY) = mnist.load_data()
    # X = X.reshape([-1, 784])
    # testX = testX.reshape([-1, 784])
    # Y= to_categorical(Y, 10)
    # testY= to_categorical(testY, 10)
    # do_dnn_1d(X,Y,testX,testY)

    (X, Y), (testX, testY) = mnist.load_data()
    X = X.reshape([-1,784])
    testX = testX.reshape([-1, 784])

    # 1d
    print (testX)
    do_svm_1d(X, Y, testX, testY)
    do_knn_1d(X, Y, testX, testY)

    # #2d
    # (X, Y), (testX, testY) = mnist.load_data()
    #
    # Y= to_categorical(Y, 10)
    # testY= to_categorical(testY, 10)
    # X = X.reshape([-1, 28, 28, 1])
    # testX = testX.reshape([-1, 28, 28, 1])
    #
    # #cnn
    # do_cnn_2d(X, Y, testX, testY)