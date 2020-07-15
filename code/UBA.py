from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import tflearn

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
import gensim
import re
from collections import namedtuple
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
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
import matplotlib.pyplot as plt

cmdlines_file="/home/liyulian/websafetyL/data/uba/MasqueradeDat/User7"
labels_file="/home/liyulian/websafetyL/data/uba/MasqueradeDat/label.txt"
word2ver_bin="uba_word2vec.bin"
max_features=300
index = 80

def get_cmdlines():
    x=np.loadtxt(cmdlines_file,dtype=str)
    x=x.reshape((150,100))
    y=np.loadtxt(labels_file, dtype=int,usecols=6)
    y=y.reshape((100, 1))
    y_train=np.zeros([50,1],int)
    y=np.concatenate([y_train,y])
    y=y.reshape((150, ))

    return x,y

def get_features_by_wordbag():
    global max_features
    global  index
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x=vectorizer.fit_transform(x)

    x_train=x[0:index,]
    x_test=x[index:,]
    y_train=y[0:index,]
    y_test=y[index:,]

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x)
    x_test = transformer.transform(x_test)
    x_train = transformer.transform(x_train)

    return x_train, x_test, y_train, y_test


def get_features_by_ngram():
    global max_features
    global  index
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    vectorizer = CountVectorizer(
                                 ngram_range=(2, 4),
                                 token_pattern=r'\b\w+\b',
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x=vectorizer.fit_transform(x)

    x_train=x[0:index,]
    x_test=x[index:,]
    y_train=y[0:index,]
    y_test=y[index:,]

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x)
    x_test = transformer.transform(x_test)
    x_train = transformer.transform(x_train)

    return x_train, x_test, y_train, y_test

def  get_features_by_wordseq():
    global max_features
    global  index
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_features,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x = np.array(list(x))

    x_train = x[0:index, ]
    x_test = x[index:, ]
    y_train = y[0:index, ]
    y_test = y[index:, ]

    return x_train, x_test, y_train, y_test

def  get_features_by_wordseq_hmm():
    global max_features
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_features,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x = np.array(list(x))

    x_train = x[0:50, ]
    x_test = x[50:, ]
    y_train = y[0:50, ]
    y_test = y[50:, ]

    return x_train, x_test, y_train, y_test


def buildWordVector(imdb_w2v,text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def  get_features_by_word2vec():
    global word2ver_bin
    global index
    global max_features

    x_all=[]

    x_arr,y=get_cmdlines()

    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    for i in range(1,30):
        filename="../data/uba/MasqueradeDat/User%d" % i
        with open(filename) as f:
            x_all.append([w.strip('\n') for w in f.readlines()])


    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print ("Find cache file %s" % word2ver_bin)
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=5, min_count=1, iter=60, workers=cores)
        model.build_vocab(x_all)
        model.train(x_all, total_examples=model.corpus_count, epochs=model.iter)
        #model.save(word2ver_bin)

    x = np.concatenate([buildWordVector(model, z, max_features) for z in x])
    x = scale(x)


    x_train = x[0:index,]
    x_test = x[index:,]

    y_train = y[0:index,]
    y_test = y[index:,]

    return x_train, x_test, y_train, y_test


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

def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_cnn(trainX, testX, trainY, testY):
    global max_features
    y_test = testY
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Building convolutional network
    inputs = Input(shape=(max_features,),name='input')
    network = Embedding(1000,128)(inputs)
    branch1 = Conv1D(128,(2,2),padding='valid',activation='relu')(network)
    branch2 = Conv1D(128,(3,3),padding='valid',activation='relu')(branch1)
    branch3 = Conv1D(128,(4,4),padding='valid',activation='relu')(branch2)
    network = Add(mode='concat', axis=1)(branch1,branch2,branch3)
    network = tf.expand_dims(network, 2)
    network = GlobalMaxPool1D()(network)
    network = Dropout(1)(network)
    predictions = Dense(2, activation='softmax')(network)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.fit(trainX, trainY, validation_data=0,shuffle=True,
              batch_size=10, epochs=10)

    y_predict_list = model.predict(testX)

    y_predict = []
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print (metrics.confusion_matrix(y_test, y_predict))

def do_rnn_wordbag(trainX, testX, trainY, testY):
    y_test=testY
    #trainX = pad_sequences(trainX, maxlen=100, value=0.)
    #testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    inputs = Input(shape=(100,))
    net = Embedding(1000, 128)(inputs)
    net = LSTM(128, dropout=0.1)(net)
    predictions = Dense(2, activation='softmax')(net)
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.005,
                             loss='categorical_crossentropy')

    # Training
    adam = optimizers.Adam(lr=0.005)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.fit(trainX, trainY, validation_split=0.1, shuffle=True,
              batch_size=10, epochs=10)

    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        #print  i[0]
        if i[0] >= 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print (metrics.confusion_matrix(y_test, y_predict))

    print (y_train)

    print ("ture")
    print (y_test)
    print ("pre")
    print (y_predict)

def do_birnn_wordbag(trainX, testX, trainY, testY):
    y_test=testY
    #trainX = pad_sequences(trainX, maxlen=100, value=0.)
    #testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    # Network building
    inputs = Input(shape=(100,))
    net = Embedding(10000, 128)(inputs)
    net = Bidirectional(LSTM(128))(net)
    net = Dropout(0.5)(net)
    predictions = Dense(2, activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.005)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.fit(trainX, trainY, validation_split=(testX, testY), shuffle=True,
              batch_size=1, epochs=10)

    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        #print  i[0]
        if i[0] >= 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print (metrics.confusion_matrix(y_test, y_predict))

def do_hmm(trainX, testX, trainY, testY):
    T=-580
    N=2
    lengths=[1]
    X=[[0]]
    print (len(trainX))
    for i in trainX:
        z=[]
        for j in i:
            z.append([j])
        #print z
        #X.append(z)
        X=np.concatenate([X,np.array(z)])
        lengths.append(len(i))

    #print lengths
    #print X.shape



    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    remodel.fit(X, lengths)

    y_predict=[]
    for i in testX:
        z=[]
        for j in i:
            z.append([j])
        y_pred=remodel.score(z)
        print (y_pred)
        if y_pred < T:
            y_predict.append(1)
        else:
            y_predict.append(0)
    y_predict=np.array(y_predict)

    print(classification_report(testY, y_predict))
    print (metrics.confusion_matrix(testY, y_predict))

    print (testY)
    print (y_predict)

def show_hmm(trainX, testX, trainY, testY):
    a=[]
    b=[]
    c=[]

    N=2
    lengths=[1]
    X=[[0]]
    print (len(trainX))
    for i in trainX:
        z=[]
        for j in i:
            z.append([j])
        X=np.concatenate([X,np.array(z)])
        lengths.append(len(i))

    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    remodel.fit(X, lengths)

    for T in range(-600,-400,10):
        y_predict = []
        for i in testX:
            z = []
            for j in i:
                z.append([j])
            y_pred = remodel.score(z)
            #print y_pred
            if y_pred < T:
                y_predict.append(1)
            else:
                y_predict.append(0)
        y_predict = np.array(y_predict)
        precision=precision_score(y_test,y_predict)
        recall=recall_score(y_test,y_predict)
        a.append(T)
        b.append(precision)
        c.append(recall)
        plt.plot(a, b,'-rD',a,c, ':g^')
        #plt.plot(a, b, 'r')
        #plt.plot(a, c, 'r')
    plt.xlabel("log probability")
    plt.ylabel("metrics.recall&precision")
    #plt.ylabel("metrics.precision")
    #plt.title("metrics.precision")
    plt.title("metrics.recall&precision")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print ("Hello uba")

    """
    print "nb and wordbag"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordbag()
    do_nb(x_train.toarray(), x_test.toarray(), y_train, y_test)
    print "xgboost and wordbag"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordbag()
    do_xgboost(x_train, x_test, y_train, y_test)
    print "xgboost and ngram"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_ngram()
    do_xgboost(x_train, x_test, y_train, y_test)
    print "mlp and wordbag"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordbag()
    do_mlp(x_train, x_test, y_train, y_test)
    print "mlp and ngram"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_ngram()
    do_mlp(x_train, x_test, y_train, y_test)
    print "xgboost and word2vec"
    max_features = 100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test = get_features_by_word2vec()
    do_xgboost(x_train, x_test, y_train, y_test)
    print "cnn and wordseq"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordseq()
    print x_train
    do_cnn(x_train, x_test, y_train, y_test)
    print "rnn and wordseq"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordseq()
    print y_train
    do_rnn_wordbag(x_train, x_test, y_train, y_test)
    print "hmm and wordseq"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordseq_hmm()
    do_hmm(x_train, x_test, y_train, y_test)
    print "rnn and wordseq"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordseq()
    #print y_train
    #print y_test
    #do_rnn_wordbag(x_train, x_test, y_train, y_test)
    show_hmm(x_train, x_test, y_train, y_test)
    do_hmm(x_train, x_test, y_train, y_test)
    """
    print ("rnn and wordseq")
    max_features=100
    print ("max_features=%d" % max_features)

    x_train, x_test, y_train, y_test=get_features_by_wordseq()
    print (x_train)
    do_rnn_wordbag(x_train, x_test, y_train, y_test)