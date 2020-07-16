from keras.layers import Input,Dropout,Dense,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import TensorBoard

# Data loading and preprocessing
from keras.datasets import mnist
import tflearn.datasets.oxflower17 as oxflower17


def cnn():

    (X,Y),(testX,testY)= mnist.load_data()

    X = X.astype('float32')
    testX = testX.astype('float32')

    Y= to_categorical(Y, 10)
    testY= to_categorical(testY, 10)

    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])

    # Building convolutional network
    inputs = Input(shape=(28, 28, 1), name='input')
    network = Conv2D(32, (3,3), activation='relu')(inputs)
    network = MaxPooling2D(pool_size=(2,2))(network)
    network = Conv2D(64, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2,2))(network)
    network = Flatten()(network)
    network = Dense(128, activation='tanh')(network)
    network = Dropout(0.5)(network)
    network = Dense(256, activation='tanh')(network)
    network = Dropout(0.5)(network)
    predictions = Dense(10, activation='softmax')(network)

    # Training
    adam = optimizers.Adam(lr=0.01)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit( X,Y,validation_split=0.1,callbacks=[TensorBoard(log_dir='./log_dir')])
    loss, accuracy = model.evaluate(testX, testY, verbose=1)
    print('loss:%.4f accuracy:%.4f' % (loss, accuracy))


def alexnet():
    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

    # Building 'AlexNet'
    inputs = Input(shape=(227, 227, 3))
    network = Conv2D(96, (11,11), strides=4, activation='relu')(inputs)
    network = MaxPooling2D(pool_size=(2,2))(network)
    network = Dropout(0.25)(network)
    network = Conv2D(256, (5,5), activation='relu')(network)
    network = MaxPooling2D(pool_size=(3,3))(network)
    network = Dropout(0.25)(network)
    network = Conv2D(384, (3,3), activation='relu')(network)
    network = Conv2D(384, (3,3), activation='relu')(network)
    network = Conv2D(256, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(3,3),strides=2)(network)
    network = Dropout(0.25)(network)
    network = Flatten()(network)
    network = Dense(4096, activation='tanh')(network)
    network = Dropout(0.5)(network)
    network = Dense(4096, activation='tanh')(network)
    network = Dropout(0.5)(network)
    predictions = Dense(17, activation='softmax')(network)

    # Training
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='momentum',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, batch_size=64)
# alexnet()

def vggnet():
    X, Y = oxflower17.load_data(resize_pics=(227, 227),one_hot=True)

    # Building 'VGG Network'
    inputs = Input(shape=(227, 227, 3))

    network = Conv2D(64, (3,3), activation='relu')(inputs)
    network = Conv2D(64, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(128, (3,3), activation='relu')(network)
    network = Conv2D(128, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(256, (3,3), activation='relu')(network)
    network = Conv2D(256, (3,3), activation='relu')(network)
    network = Conv2D(256, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2,2),strides=2)(network)

    network = Flatten()(network)
    network = Dense(4096, activation='relu')(network)
    network = Dropout(0.5)(network)
    network =  Dense(4096, activation='relu')(network)
    network = Dropout(0.5)(network)
    predictions = Dense(17, activation='softmax')(network)

    # Training
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, Y, n_epoch=500, batch_size=32,validation_split=0.2)

cnn()
# vggnet()