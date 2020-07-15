import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
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

import tflearn

from sklearn.neural_network import MLPClassifier
from tensorflow.contrib import learn
import subprocess
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing
import codecs

max_features=10000
max_document_length=100
min_opcode_count=2


#pro
#webshell_dir="../data/webshell/b/"
#whitefile_dir="../data/webshell/w/"
webshell_dir="/home/liyulian/websafetyL/data/webshell/webshell/PHP/"
whitefile_dir="/home/liyulian/websafetyL/data/webshell/normal/php/"
#webshell_dir="../data/webshell/dev-b/"
#whitefile_dir="../data/webshell/dev-w/"
check_dir="/home/liyulian/websafetyL/data/"
white_count=0
black_count=0
php_bin="/home/liyulian/websafetyL/data/webshell/webshell"
#php_bin="/Users/maidou/Desktop/book/2book/2book/opt/php/bin/php"
#php_bin=" /home/fuxi/dev/opt/php/bin/php"


pkl_file="webshell-opcode-cnn.pkl"

data_pkl_file="data-webshell-opcode-tf.pkl"
label_pkl_file="label-webshell-opcode-tf.pkl"


#pro
#php_bin="/home/fuxi/dev/opt/php/bin/php"


def do_xgboost(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    print ("xgboost")
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') or filename.endswith('.txt'):
                fulepath = os.path.join(path, filename)
                print ("Load %s" % fulepath)
                t = load_file(fulepath)
                files_list.append(t)

    return files_list

def load_files_opcode_re(dir):
    global min_opcode_count
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') :
                fulepath = os.path.join(path, filename)
                print ("Load %s opcode" % fulepath)
                t = load_file_opcode(fulepath)
                if len(t) > min_opcode_count:
                    files_list.append(t)
                else:
                    print ("Load %s opcode failed" % fulepath)
                #print "Add opcode %s" % t

    return files_list


def load_file(file_path):
    t=""
    with codecs.open(file_path,encoding=u'utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            t+=line
    return t

def load_file_opcode(file_path):
    global php_bin
    t=""
    cmd=php_bin+" -dvld.active=1 -dvld.execute=0 "+file_path
    #print "exec "+cmd
    status ,output=subprocess.getstatusoutput(cmd)

    t=output
        #print t

    tokens=re.findall(r'\s(\b[A-Z_]+\b)\s',output)
    t=" ".join(tokens)

    print ("opcode count %d" % len(t))
    return t



def load_files(path):
    files_list=[]
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path=path+file
                print ("Load %s" % file_path)
                t=load_file(file_path)
                files_list.append(t)
    return  files_list

def get_feature_by_bag_tfidf():
    global white_count
    global black_count
    global max_features
    print ("max_features=%d" % max_features)
    x=[]
    y=[]

    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)

    x=webshell_files_list+wp_files_list
    y=y1+y2

    CV = CountVectorizer(ngram_range=(2, 4), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x)
    x = x_tfidf.toarray()

    return x,y

def get_feature_by_opcode():
    global white_count
    global black_count
    global max_features
    global webshell_dir
    global whitefile_dir
    print ("max_features=%d webshell_dir=%s whitefile_dir=%s" % (max_features,webshell_dir,whitefile_dir))
    x=[]
    y=[]

    webshell_files_list = load_files_opcode_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_opcode_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)


    x=webshell_files_list+wp_files_list
    #print x
    y=y1+y2

    CV = CountVectorizer(ngram_range=(2, 4), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)

    x=CV.fit_transform(x).toarray()

    return x,y


def get_feature_by_opcode_tf():
    global white_count
    global black_count
    global max_document_length
    x=[]
    y=[]

    if os.path.exists(data_pkl_file) and os.path.exists(label_pkl_file):
        f = open(data_pkl_file, 'rb')
        x = pickle.load(f)
        f.close()
        f = open(label_pkl_file, 'rb')
        y = pickle.load(f)
        f.close()
    else:
        webshell_files_list = load_files_opcode_re(webshell_dir)
        y1=[1]*len(webshell_files_list)
        black_count=len(webshell_files_list)

        wp_files_list =load_files_opcode_re(whitefile_dir)
        y2=[0]*len(wp_files_list)

        white_count=len(wp_files_list)


        x=webshell_files_list+wp_files_list
        #print x
        y=y1+y2

        vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                                  min_frequency=0,
                                                  vocabulary=None,
                                                  tokenizer_fn=None)
        x=vp.fit_transform(x, unused_y=None)
        x=np.array(list(x))

        f = open(data_pkl_file, 'wb')
        pickle.dump(x, f)
        f.close()
        f = open(label_pkl_file, 'wb')
        pickle.dump(y, f)
        f.close()
    #print x
    #print y
    return x,y



def  get_features_by_tf():
    global  max_document_length
    global white_count
    global black_count
    x=[]
    y=[]

    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)


    x=webshell_files_list+wp_files_list
    y=y1+y2

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x=np.array(list(x))
    return x,y

def check_webshell(clf,dir):
    all=0
    all_php=0
    webshell=0

    webshell_files_list = load_files_re(webshell_dir)

    transformer = TfidfVectorizer(ngram_range=(3, 3),
                                 token_pattern=r'\b\d+\b',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)


    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            fulepath=os.path.join(path, filename)
            t = load_file(fulepath)
            t_list=[]
            t_list.append(t)
            x2 = transformer.transform(t).toarray()
            y_pred = clf.predict(x2)
            all+=1
            if filename.endswith('.php'):
                all_php+=1
            if y_pred[0] == 1:
                print ("%s is webshell" % fulepath)
                webshell+=1

    print ("Scan %d files(%d php files),%d files is webshell" %(all,all_php,webshell))


def do_check(x,y,clf):
    clf.fit(x, y)
    print ("check_webshell")
    #check_webshell(clf,"../data/webshell/normal/php/")
    #/Users/maidou/Downloads/webshell-master/php
    check_webshell(clf,check_dir)



def do_metrics(y_test,y_pred):
    print ("metrics.accuracy_score:")
    print (metrics.accuracy_score(y_test, y_pred))
    print ("metrics.confusion_matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))
    print ("metrics.precision_score:")
    print (metrics.precision_score(y_test, y_pred))
    print ("metrics.recall_score:")
    print (metrics.recall_score(y_test, y_pred))
    print ("metrics.f1_score:")
    print (metrics.f1_score(y_test,y_pred))

def do_mlp(x,y):
    #mlp
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)

    #print clf
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (y_train)
    print (y_pred)
    print (y_test)
    do_metrics(y_test,y_pred)

def do_nb(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test,y_pred)

def do_svm(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    do_metrics(y_test,y_pred)

def do_cnn(x,y):
    global max_document_length
    print ("CNN and tf")
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Building convolutional network
    inputs = Input(shape=(max_document_length,),name='input')
    network = Embedding(1000000,128)(inputs)
    branch1 = Conv1D(128, (3,3), padding='valid', activation='relu')(network)
    branch2 = Conv1D(128, (4,4), padding='valid', activation='relu')(branch1)
    branch3 = Conv1D(128, (5,5), padding='valid', activation='relu')(branch2)
    network = Add(mode='concat', axis=1)(branch1, branch2, branch3)
    network = tf.expand_dims(network, 2)
    network = GlobalMaxPool1D()(network)
    network = Dropout(0.8)(network)
    predictions = Dense(2, activation='softmax')(network)

    model = tflearn.DNN(network, tensorboard_verbose=0)
    #if not os.path.exists(pkl_file):
    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX, trainY,batch_size=100,
                  epochs=5, shuffle=True, validation_split=0.1)
    #    model.save(pkl_file)
    #else:
    #    model.load(pkl_file)

    y_predict_list=model.predict(testX)
    #y_predict = list(model.predict(testX,as_iterable=True))

    y_predict=[]
    for i in y_predict_list:
        print (i[0])
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)
    print ('y_predict_list:')
    print (y_predict_list)
    print ('y_predict:')
    print (y_predict)
    #print  y_test

    do_metrics(y_test, y_predict)


def do_rnn(x,y):
    global max_document_length
    print ("RNN")
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    inputs = Input(shape=(max_document_length,))
    net = Embedding(10240000,128)(inputs)(inputs)
    net = LSTM(128,dropout=0.8)(net)
    predictions = Dense(2,activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.fit(trainX,trainY,validation_split=0.1,batch_size=10,epochs=5)

    y_predict_list=model.predict(testX)
    y_predict=[]
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    do_metrics(y_test, y_predict)

def do_rf(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    do_metrics(y_test,y_pred)

if __name__ == '__main__':
    #x, y = get_feature_by_opcode_tf()
    #x,y=get_feature_by_bag_tfidf()
    #x, y = get_feature_by_opcode()
    #print "load %d white %d black" % ( white_count,black_count )

    #mlp
    #do_mlp(x,y)
    #nb
    #do_nb(x,y)
    #do_rf(x,y)
    #svm
    #do_svm(x,y)
    #do_check(x,y,clf)

    #x,y=get_features_by_tf()

    #do_cnn(x,y)
    #do_rnn(x,y)



    print ("xgboost and bag and 2-gram")
    max_features=5000
    print ("max_features=%d" % max_features)
    x, y = get_feature_by_bag_tfidf()
    print ("load %d white %d black" % (white_count, black_count))
    do_xgboost(x, y)

    # print ("xgboost and opcode and 4-gram")
    # max_features=10000
    # max_document_length=4000
    # print ("max_features=%d max_document_length=%d" % (max_features,max_document_length))
    # x, y = get_feature_by_opcode()
    # print ("load %d white %d black" % (white_count, black_count))
    # do_xgboost(x, y)

    #
    # print ("xgboost and wordbag and 2-gram")
    # max_features=10000
    # max_document_length=4000
    # print ("max_features=%d max_document_length=%d" % (max_features,max_document_length))
    # x, y = get_feature_by_bag_tfidf()
    # print ("load %d white %d black" % (white_count, black_count))
    # do_xgboost(x, y)