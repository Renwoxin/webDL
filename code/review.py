from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.layers import Input,Dropout,Dense,Embedding,LSTM,Flatten
from keras.layers import  Conv1D,GlobalMaxPool1D
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
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
import tflearn

max_features=400
max_document_length=1000
vocabulary=None
doc2ver_bin="doc2ver.bin"
word2ver_bin="word2ver.bin"
#LabeledSentence = gensim.models.doc2vec.LabeledSentence
SentimentDocument = namedtuple('SentimentDocument', 'words tags')




def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    f.close()
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
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    path="/home/liyulian/websafetyL/data/review/aclImdb/train/pos/"
    print ("Load %s" % path)
    x_train=load_files_from_dir(path)
    y_train=[0]*len(x_train)
    path="/home/liyulian/websafetyL/data/review/aclImdb/train/neg/"
    print ("Load %s" % path)
    tmp=load_files_from_dir(path)
    y_train+=[1]*len(tmp)
    x_train+=tmp

    path="../data/review/aclImdb/test/pos/"
    print ("Load %s" % path)
    x_test=load_files_from_dir(path)
    y_test=[0]*len(x_test)
    path="../data/review/aclImdb/test/neg/"
    print ("Load %s" % path)
    tmp=load_files_from_dir(path)
    y_test+=[1]*len(tmp)
    x_test+=tmp

    return x_train, x_test, y_train, y_test

def get_features_by_wordbag():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test

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

def do_nb_doc2vec(x_train, x_test, y_train, y_test):
    print ("NB and doc2vec")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print ("SVM and wordbag")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


def do_svm_doc2vec(x_train, x_test, y_train, y_test):
    print ("SVM and doc2vec")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    print ("rf and doc2vec")
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


def get_features_by_wordbag_tfidf():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 binary=True)
    print (vectorizer)
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,binary=True,
                                 min_df=1 )
    print (vectorizer)
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train=transformer.fit_transform(x_train)
    x_train=x_train.toarray()
    x_test=transformer.transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test

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
    network = Embedding(1000000,128)(inputs)
    branch1 = Conv1D(128, (3,3), padding='valid', activation='relu')(network)
    branch2 =Conv1D(128, (4,4), padding='valid', activation='relu')(branch1)
    branch3 = Conv1D(128, (5,5), padding='valid', activation='relu')(branch2)
    network = Add(mode='concat', axis=1)(branch1, branch2, branch3)
    network = tf.expand_dims(network, 2)
    network = GlobalMaxPool1D()(network)
    network = Dropout(0.8)(network)
    predictions = Dense(2, activation='softmax')(network)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX, trainY,epochs=5, shuffle=True,
              validation_data=(testX, testY),batch_size=100)


def do_cnn_doc2vec(trainX, testX, trainY, testY):
    global max_features
    print ("CNN and doc2vec")

    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Building convolutional network
    inputs = Input(shape=(max_features,), name='input')
    network = Embedding(1000000, 128,validate_indices=False)(inputs)
    branch1 = Conv1D(128, 3, padding='valid', activation='relu')(network)
    branch2 = Conv1D(128, 4, padding='valid', activation='relu')(network)
    branch3 = Conv1D(128, 5, padding='valid', activation='relu')(network)
    network = Add(mode='concat', axis=1)(branch1, branch2, branch3)
    network = tf.expand_dims(network, 2)
    network = GlobalMaxPool1D()(network)
    network = Dropout(0.8)(network)
    predictions = Dense(2, activation='softmax')(network)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.fit(trainX, trainY,epochs=5, shuffle=True,
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
    inputs = Input(shape=(max_document_length,))
    net = Embedding(10240000,128)(inputs)
    net = LSTM(128, dropout=0.8)(net)
    predictions = Dense(2, activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=10,epochs=5)


def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    print ("MLP and wordbag")

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

def do_dnn_doc2vec(x_train, x_test, y_train, y_test):
    print ("MLP and doc2vec")
    global max_features
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
    x_train, x_test, y_train, y_test=load_all_files()

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x_train=vp.fit_transform(x_train, unused_y=None)
    x_train=np.array(list(x_train))

    x_test=vp.transform(x_test)
    x_test=np.array(list(x_test))
    return x_train, x_test, y_train, y_test


def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        #labelized.append(LabeledSentence(v, [label]))
        #labelized.append(LabeledSentence(words=v,tags=label))
        labelized.append(SentimentDocument(v, [label]))
    return labelized

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs),dtype='float')


def getVecsByWord2Vec(model, corpus, size):
    global max_document_length
    #x=np.zeros((max_document_length,size),dtype=float, order='C')
    x=[]

    for text in corpus:
        xx = []
        for i, vv in enumerate(text):
            try:
                xx.append(model[vv].reshape((1,size)))
            except KeyError:
                continue

        x = np.concatenate(xx)

    x=np.array(x, dtype='float')
    return x


def  get_features_by_doc2vec():
    global  max_features
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    x=x_train+x_test
    cores=multiprocessing.cpu_count()
    #models = [
        # PV-DBOW
    #    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
        # PV-DM w/average
    #    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    #]
    if os.path.exists(doc2ver_bin):
        print("Find cache file %s" % doc2ver_bin)
        model=Doc2Vec.load(doc2ver_bin)
    else:
        model=Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores,iter=60)


        #for model in models:
        #    model.build_vocab(x)
        model.build_vocab(x)

        #models[1].reset_from(models[0])

        #for model in models:
        #    model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        #models[0].train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    #x_test=getVecs(models[0],x_test,max_features)
    #x_train=getVecs(models[0],x_train,max_features)
    x_test=getVecs(model,x_test,max_features)
    x_train=getVecs(model,x_train,max_features)

    return x_train, x_test, y_train, y_test

def  get_features_by_word2vec():
    global  max_features
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x=x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print ("Find cache file %s" % word2ver_bin)
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=5, min_count=10, iter=10, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)


    x_train=getVecsByWord2Vec(model,x_train,max_features)
    x_test = getVecsByWord2Vec(model, x_test, max_features)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    print ("Hello review")
    #print "get_features_by_wordbag_tfidf"
    #x_train, x_test, y_train, y_test=get_features_by_wordbag_tfidf()
    #print "get_features_by_tf"
    #x_train, x_test, y_train, y_test=get_features_by_tf()
    #NB
    #do_nb_wordbag(x_train, x_test, y_train, y_test)
    #SVM
    #do_svm_wordbag(x_train, x_test, y_train, y_test)

    #print "get_features_by_wordbag_tfidf"
    #x,y=get_features_by_wordbag_tfidf()
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

    #show_diffrent_max_features()



    #DNN
    #do_dnn_wordbag(x_train, x_test, y_train, y_test)

    #print "get_features_by_tf"
    #x,y=get_features_by_tf()
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    #CNN
    #do_cnn_wordbag(x_train, x_test, y_train, y_test)


    #RNN
    #do_rnn_wordbag(x_train, x_test, y_train, y_test)
    #print "get_features_by_doc2vec"
    x_train, x_test, y_train, y_test=get_features_by_doc2vec()
    #print "get_features_by_word2vec"
    #x_train, x_test, y_train, y_test=get_features_by_word2vec()
    #print x_train
    #print x_test

    #NB
    do_nb_doc2vec(x_train, x_test, y_train, y_test)
    #CNN
    #DNN
    # do_dnn_doc2vec(x_train, x_test, y_train, y_test)
    #SVM
    # do_svm_doc2vec(x_train, x_test, y_train, y_test)
    # do_rf_doc2vec(x_train, x_test, y_train, y_test)