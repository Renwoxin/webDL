import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf

from keras.layers import Input, Dropout, Dense
from keras.layers import Embedding,LSTM,Bidirectional
from keras.layers import Conv1D, GlobalMaxPool1D
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Add,BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

from sklearn.neural_network import MLPClassifier
from tensorflow.contrib import learn
import gensim
import re
from collections import namedtuple
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing
from hmmlearn import hmm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

dga_file="/home/liyulian/websafetyL/data/dga/dga.txt"
alexa_file="/home/liyulian/websafetyL/data/dga/top-1m.csv"

def load_alexa():
    x=[]
    data = pd.read_csv(alexa_file, sep=",",header=None)
    x=[i[1] for i in data.values]
    return x

def load_dga():
    x=[]
    data = pd.read_csv(dga_file, sep="\t", header=None,
                      skiprows=18)
    x=[i[1] for i in data.values]
    return x

def get_feature_charseq():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    t=[]
    for i in x:
        v=[]
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)

    x=t
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train, x_test, y_train, y_test


def get_aeiou(domain):
    count = len(re.findall(r'[aeiou]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):
    count=len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_feature():
    from sklearn import preprocessing
    alexa=load_alexa()
    dga=load_dga()
    v=alexa+dga
    y=[0]*len(alexa)+[1]*len(dga)
    x=[]

    for vv in v:
        vvv=[get_aeiou(vv),get_uniq_char_num(vv),get_uniq_num_num(vv),len(vv)]
        x.append(vvv)

    x=preprocessing.scale(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)
    return x_train, x_test, y_train, y_test

def get_feature_2gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 2),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test


def get_feature_234gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 4),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test

def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_mlp(x_train, x_test, y_train, y_test):

    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_rnn(trainX, testX, trainY, testY):
    max_document_length=64
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    inputs = Input(shape=(max_document_length,))
    net = Embedding(10240000,64)(inputs)
    net = LSTM(64,dropout=0.1)(net)
    predictions = Dense(2, activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=10,epochs=1)
    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        print  (i[0])
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print (metrics.confusion_matrix(y_test, y_predict))

if __name__ == "__main__":
    print ("Hello dga")
    print ("234-gram & mlp")
    x_train, x_test, y_train, y_test = get_feature_234gram()
    do_mlp(x_train, x_test, y_train, y_test)
    """
    print "text feature & nb"
    x_train, x_test, y_train, y_test = get_feature()
    do_nb(x_train, x_test, y_train, y_test)
    print "text feature & xgboost"
    x_train, x_test, y_train, y_test = get_feature()
    do_xgboost(x_train, x_test, y_train, y_test)
    print "text feature & mlp"
    x_train, x_test, y_train, y_test = get_feature()
    do_mlp(x_train, x_test, y_train, y_test)
    print "charseq & rnn"
    x_train, x_test, y_train, y_test = get_feature_charseq()
    do_rnn(x_train, x_test, y_train, y_test)
    print "2-gram & mlp"
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_mlp(x_train, x_test, y_train, y_test)
    print "2-gram & XGBoost"
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_xgboost(x_train, x_test, y_train, y_test)
    print "2-gram & nb"
    x_train, x_test, y_train, y_test=get_feature_2gram()
    do_nb(x_train, x_test, y_train, y_test)
"""