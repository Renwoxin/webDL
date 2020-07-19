# -*- coding: utf-8 -*-
"""
    This is a demo of credit_card_fraud, where  Credit Card Fraud Detection is used
"""

# data preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Training
from sklearn import metrics
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def get_feature(path):
    """

    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        X_test: DataFrame,shape=(x.shape[0] * test_size,x.shape[1]), the features for testing
        Y_train: list:x.shape[0] * (1-test_size), the labels for training
        Y_test: list:x.shape[0] * test_size, the labels for testing

    """

    df = pd.read_csv(path)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y=df['Class']
    #print y
    features = df.drop(['Class'], axis=1).columns
    x=df[features]
    #print x

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)
    return X_train, X_test, Y_train, Y_test



def get_feature_undersampling(path):
    """

    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        X_test: DataFrame, shape(x_index*test_size, feature_number(feature_map)), the features for testing
        Y_train: list:x_index*(1-test_size), the labels for training
        Y_test: list:x_index*test_size, the labels for testing

    """

    df = pd.read_csv(path)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time','Amount'], axis=1)

    number_fraud = len(df[df.Class==1])
    fraud_index = np.array(df[df.Class==1].index)

    normal_index = df[df.Class==0].index
    radom_choice_normal_index = np.random.choice(normal_index,size=number_fraud,replace=False)

    x_index = np.concatenate([fraud_index,radom_choice_normal_index])
    df = df.drop(['Class'],axis=1)

    x = df.iloc[x_index,:]

    y = [1] * number_fraud + [0] * number_fraud
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)
    return X_train, X_test, Y_train, Y_test


def get_feature_undersampling_2(path):
    """

    Args:
            path: the path of Dataset

        Returns:
            X_train_1: DataFrame, the features for training
            X_test: DataFrame, the features for testing
            Y_train_1: list, the labels for training
            Y_test: list, the labels for testing

    """

    df = pd.read_csv(path)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y = df['Class']
    features = df.drop(['Class'], axis=1).columns
    x = df[features]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)

    print ("raw data")
    print (pd.value_counts(Y_train))

    number_fraud=len(Y_train[Y_train==1])
    print (number_fraud)
    fraud_index=np.array(Y_train[Y_train==1].index)
    print (fraud_index)

    normal_index=Y_train[Y_train==0].index
    random_choice_index=np.random.choice(normal_index,size=number_fraud,replace=False)

    x_index=np.concatenate([fraud_index,random_choice_index])
    print (x_index)
    #df = df.drop(['Class'], axis=1)
    X_train_1=x.iloc[x_index,:]

    #print x
    Y_train_1=[1]*number_fraud+[0]*number_fraud

    print ("Undersampling data")
    print (pd.value_counts(Y_train_1))

    return X_train_1, X_test, Y_train_1, Y_test

def get_feature_upsampling(path):
    """

    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, the features for training
        X_test: DataFrame, the features for testing
        Y_train: list, the labels for training
        Y_test: list, the labels for testing

    """
    df = pd.read_csv(path)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y = df['Class']
    features = df.drop(['Class'], axis=1).columns
    x = df[features]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)

    print ("raw data")
    print (pd.value_counts(Y_train))

    os = SMOTE(random_state=0)
    X_train_1,Y_train_1=os.fit_sample(X_train,Y_train)
    print ("Smote data")
    print (pd.value_counts(Y_train_1))


    return X_train, X_test, Y_train, Y_test


def do_xgboost(X_train, X_test, Y_train, Y_test):
    """

    Args:
        X_train:DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        X_test: DataFrame, shape(x_index*test_size, feature_number(feature_map)), the features for testing
        Y_train: list, the labels for training
        Y_test: list, the labels for testing

    Returns:None

    """
    xgb_model = xgb.XGBClassifier().fit(X_train, Y_train)
    Y_pred = xgb_model.predict(X_test)
    do_metrics(Y_test, Y_pred)


def do_mlp(x_train, x_test, y_train, y_test):
    #mlp
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)

def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test,y_pred)


def do_metrics(Y_test,Y_pred):
    """

    Args:
        Y_test: list, the labels for testing
        Y_pred: list, the labels for prediction

    Returns:None

    """
    print ("metrics.accuracy_score:")
    print (metrics.accuracy_score(Y_test, Y_pred))
    print ("metrics.confusion_matrix:")
    print (metrics.confusion_matrix(Y_test, Y_pred))
    print ("metrics.precision_score:")
    print (metrics.precision_score(Y_test, Y_pred))
    print ("metrics.recall_score:")
    print (metrics.recall_score(Y_test, Y_pred))
    print ("metrics.f1_score:")
    print (metrics.f1_score(Y_test,Y_pred))


def run_1(path):
    """

    Args:
        path: the path of Dataset

    Returns: None

    """
    x_train, x_test, y_train, y_test=get_feature(path)
    do_xgboost(x_train, x_test, y_train, y_test)
    do_mlp(x_train, x_test, y_train, y_test)
    do_nb(x_train, x_test, y_train, y_test)


def run_2(path):
   """

   Args:
       path: the path of Dataset

   Returns:None

   """

   x_train, x_test, y_train, y_test = get_feature_undersampling(path)
   print("XGBoost")
   do_xgboost(x_train, x_test, y_train, y_test)
   print("mlp")
   do_mlp(x_train, x_test, y_train, y_test)
   print("nb")
   do_nb(x_train, x_test, y_train, y_test)



def run_3(path):
    """

    Args:
        path: the path of Dataset

    Returns: None

    """
    x_train, x_test, y_train, y_test=get_feature_upsampling(path)
    print ("XGBoost")
    do_xgboost(x_train, x_test, y_train, y_test)
    print ("mlp")
    do_mlp(x_train, x_test, y_train, y_test)
    print ("nb")
    do_nb(x_train, x_test, y_train, y_test)


if __name__=="__main__":
    path = "/home/liyulian/code/websafetyL/data/fraud/creditcard.csv"
    # 特征提取使用标准化
    # run_1(path)
    # 特征提取使用标准化&降采样
    run_2(path)
    # 特征提取使用标准化&过采样
    # run_3(path)

