# data preprocessing
from keras.datasets import mnist
from keras.utils import to_categorical
from tflearn.datasets import oxflower17

# Building Network
from keras.layers import Input,Conv2D,MaxPool2D,Flatten,Dense
from keras.layers import Dropout,BatchNormalization
from keras import regularizers

# Training
from keras import optimizers
from keras.models import Model


def cnn():
    """

    Returns:None

    """

    # data preprocessing
    (train_X,train_Y),(test_X,test_Y) = mnist.load_data()

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    train_Y = to_categorical(train_Y,num_classes=10)
    test_Y = to_categorical(test_Y,num_classes=10)

    train_X = train_X.reshape([-1,28,28,1])
    test_X = test_X.reshape([-1,28,28,1])

    # Building 'CNN' Network
    inputs = Input(shape=(28,28,1,),name='inputs')
    network = Conv2D(32,(3,3),activation='relu',kernel_regularizer=regularizers.l2())(inputs)
    network = MaxPool2D(pool_size=(2,2))(network)
    network = BatchNormalization()(network)
    network = Conv2D(64,(3,3),activation='relu',kernel_regularizer=regularizers.l2())(network)
    network = MaxPool2D(pool_size=(2,2))(network)
    network = BatchNormalization()(network)
    network = Flatten()(network)
    network = Dense(128,activation='tanh')(network)
    network = Dropout(0.8)(network)
    network = Dense(256,activation='tanh')(network)
    network = Dropout(0.8)(network)
    predictions = Dense(10,activation='softmax')(network)

    # Training
    adam = optimizers.Adam(lr=0.01)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_X,train_Y,epochs=10,validation_split=0.2)
    loss,accuracy = model.evaluate(test_X,test_Y,verbose=1)
    print('loss:%.4f accuracy: %.4f' % (loss,accuracy))


def alexnet():
    """

    Returns:None

    """
    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

    # Building 'AlexNet'
    inputs = Input(shape=(227, 227, 3))
    network = Conv2D(96, (11,11), strides=4, activation='relu')(inputs)
    network = MaxPool2D(pool_size=(2,2))(network)
    network = BatchNormalization()(network)
    network = Dropout(0.25)(network)
    network = Conv2D(256, (5,5), activation='relu')(network)
    network = MaxPool2D(pool_size=(3,3))(network)
    network = BatchNormalization()(network)
    network = Dropout(0.25)(network)
    network = Conv2D(384, (3,3), activation='relu')(network)
    network = Conv2D(384, (3,3), activation='relu')(network)
    network = Conv2D(256, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(3,3),strides=2)(network)
    network = BatchNormalization()(network)
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
    model.fit(X, Y, epochs=1000, validation_set=0.1, batch_size=64)


def vggnet():
    """

    Returns:None

    """
    X, Y = oxflower17.load_data(resize_pics=(227, 227),one_hot=True)

    # Building 'VGG Network'
    inputs = Input(shape=(227, 227, 3))

    network = Conv2D(64, (3,3), activation='relu')(inputs)
    network = Conv2D(64, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(128, (3,3), activation='relu')(network)
    network = Conv2D(128, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(256, (3,3), activation='relu')(network)
    network = Conv2D(256, (3,3), activation='relu')(network)
    network = Conv2D(256, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(2,2),strides=2)(network)

    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = Conv2D(512, (3,3), activation='relu')(network)
    network = MaxPool2D(pool_size=(2,2),strides=2)(network)

    network = Flatten()(network)

    network = Dense(4096, activation='relu')(network)
    network = Dropout(0.5)(network)
    network = Dense(4096, activation='relu')(network)
    network = Dropout(0.5)(network)
    predictions = Dense(17, activation='softmax')(network)

    # Training
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, Y, n_epoch=500, batch_size=32,validation_split=0.2)



# cnn()
# alexnet()
vggnet()

