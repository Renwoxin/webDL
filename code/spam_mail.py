import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

from keras.layers import Input,Dropout,Dense,Embedding,LSTM
from keras.layers import Conv1D,GlobalMaxPool1D
from keras.layers import Conv2D,MaxPool2D,Add,BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tflearn
import codecs

import tensorflow as tf
max_features=5000
max_document_length=100


def load_one_file(filename):
    x=""
    with codecs.open(filename,encoding=u'utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    return x

def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,2):
        path="/home/liyulian/websafetyL/data/mail/enron%d/ham" % i
        print ("Load %s" % path)
        ham+=load_files_from_dir(path)
        path="/home/liyulian/websafetyL/data/mail/enron%d/spam" % i
        print ("Load %s" % path)
        spam+=load_files_from_dir(path)
    return ham,spam

def get_features_by_wordbag():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    return x,y

def show_diffrent_max_features():
    global max_features
    a=[]
    b=[]
    for i in range(1000,20000,2000):
        max_features=i
        print ("max_features=%d" % i)
        x, y = get_features_by_wordbag()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        score=metrics.accuracy_score(y_test, y_pred)
        a.append(max_features)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_features")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_features")
    plt.legend()
    plt.show()

def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print ("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print ("SVM and wordbag")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def get_features_by_wordbag_tfidf():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(binary=False,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    print (transformer)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return  x,y


def do_cnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print ("CNN and tf")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Building convolutional network
    inputs = Input(shape=(max_document_length,), name='input')
    network = Embedding(1000000, 128)(inputs)
    branch1 = Conv1D(128, (3,3), padding='valid', activation='relu')(network)
    branch2 = Conv1D(128, (4,4), padding='valid', activation='relu')(branch1)
    branch3 = Conv1D(128, (5,5), padding='valid', activation='relu')(branch2)
    network = Add(mode='concat', axis=1)(branch1,branch2,branch3)
    network = tf.expand_dims(network, 2)
    network = GlobalMaxPool1D()(network)
    network = Dropout(0.8)(network)
    predictions = Dense(2, activation='softmax')(network)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True,
              validation_data=(testX, testY),batch_size=100)

def do_rnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print ("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    inputs = Input((max_document_length,))
    net = Embedding(10240000,128)(inputs)
    net = LSTM(128, dropout=0.8)(net)
    predictions = Dense(2, activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=10,epochs=5)


def do_dnn_wordbag(x_train, x_test, y_train, y_testY):
    print ("DNN and wordbag")

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))



def  get_features_by_tf():
    global  max_document_length
    x=[]
    y=[]
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x=np.array(list(x))
    return x,y



if __name__ == "__main__":
    print ("Hello spam-mail")
    print ("get_features_by_wordbag")
    x,y=get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    do_svm_wordbag(x_train, x_test, y_train, y_test)
    # do_nb_wordbag(x_train, x_test, y_train, y_test)
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)

    print ("get_features_by_wordbag_tfidf")
    x,y=get_features_by_wordbag_tfidf()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    do_svm_wordbag(x_train, x_test, y_train, y_test)
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)
    #NB
    # do_nb_wordbag(x_train, x_test, y_train, y_test)
    #show_diffrent_max_features()

    #SVM
    #do_svm_wordbag(x_train, x_test, y_train, y_test)

    # DNN
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)

    # print ("get_features_by_tf")
    # x,y=get_features_by_tf()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    #CNN
    #do_cnn_wordbag(x_train, x_test, y_train, y_test)


    #RNN
    #do_rnn_wordbag(x_train, x_test, y_train, y_test)