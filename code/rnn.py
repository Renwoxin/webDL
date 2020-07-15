from __future__ import division, print_function, absolute_import
import os
import keras
import pickle
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Embedding,Dropout
from keras.layers import Dense, LSTM, Input,Bidirectional
from keras import optimizers

from six.moves import urllib
import tflearn

# IMDB Dataset loading
# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                 valid_portion=0.1)
(trainX, trainY), (testX, testY) = imdb.load_data(path='imdb.npz', num_words=10000)


def lstm(trainX, trainY,testX, testY):
    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    inputs = Input(shape=(100,),name='inputs')
    net = Embedding(10000,128)(inputs)
    net = LSTM(128, dropout=0.8)(net)
    predictions = Dense(2, activation='softmax')(net)

    # Training
    adam = optimizers.adam(lr=0.001)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32)

def bi_lstm(trainX, trainY,testX, testY):
    trainX = pad_sequences(trainX, maxlen=200, value=0.)
    testX = pad_sequences(testX, maxlen=200, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY,num_classes=2)
    testY = to_categorical(testY,num_classes=2)

    # Network building
    inputs = Input(shape=(200,))
    net = Embedding(20000, 128)(inputs)
    net = Bidirectional(LSTM(128))(net)
    net = Dropout(0.5)(net)
    predictions = Dense(2, activation='softmax')(net)

    # Training
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(trainX, trainY, validation_split=0.1,batch_size=64)

def shakespeare():


    path = "shakespeare_input.txt"
    #path = "shakespeare_input-100.txt"
    char_idx_file = 'char_idx.pickle'

    if not os.path.isfile(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt", path)

    maxlen = 25

    char_idx = None
    if os.path.isfile(char_idx_file):
        print('Loading previous char_idx')
        char_idx = pickle.load(open(char_idx_file, 'rb'))

    X, Y, char_idx = \
        textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                             pre_defined_char_idx=char_idx)

    pickle.dump(char_idx, open(char_idx_file, 'wb'))

    inputs = Input([None, maxlen, len(char_idx)])
    g = LSTM(512, return_seq=True)(inputs)
    g = Dropout(0.5)(g)
    g = LSTM(512, return_seq=True)(g)
    g = Dropout(0.5)(g)
    g = LSTM(512)(g)
    g = Dropout(0.5)(g)
    predictions = Dense(len(char_idx), activation='softmax')(g)

    adam = optimizers.adam(lr=0.001)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path='model_shakespeare')

    for i in range(50):
        seed = random_sequence_from_textfile(path, maxlen)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='shakespeare')
        print("-- TESTING...")
        print("-- Test with temperature of 1.0 --")
        print(m.generate(600, temperature=1.0, seq_seed=seed))
        #print(m.generate(10, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(600, temperature=0.5, seq_seed=seed))

lstm(trainX, trainY,testX, testY)
# bi_lstm(trainX, trainY,testX, testY)
# shakespeare()