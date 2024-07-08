import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import validation_curve


random.seed(8)



data_path="./data/HAPTDataSet/"


def readdata(datapath):
    X_train = pd.read_table(os.path.join(datapath,'Train/X_train.txt'), header=None, sep=" ")
    y_train = pd.read_table(os.path.join(datapath,'Train/y_train.txt'), header = None, sep = " ")
    X_test = pd.read_table(os.path.join(datapath,'Test/X_test.txt'), header=None, sep=" ")
    y_test = pd.read_table(os.path.join(datapath,'Test/y_test.txt'), header=None, sep=" ")
    return X_train,y_train,X_test,y_test


def cross_validate_linear_svm(train_feature,train_labels,param_name = "C", param_range = np.logspace(-2, 2, 10), cv = 5, scoring = "accuracy", n_jobs = 6, random_state = 8):
    # Declare the classfier
    clf_svc = LinearSVC(random_state = random_state)
    train_scores, val_scores = validation_curve(clf_svc,train_feature,train_labels,param_name = param_name ,param_range = param_range ,cv = cv ,scoring = scoring ,n_jobs = n_jobs)
    return train_scores,val_scores

def test_read_and_validate():
    X_train,y_train,X_test,y_test = readdata(data_path)
    print(X_train.iloc[:10, :10].head())
    print(y_train.iloc[:100,:])
    C_params = np.logspace(-6, 3, 10)
    cross_validate_linear_svm(X_train.values,y_train.values.flatten(),param_name = "C",param_range = C_params, cv = 5, scoring = "accuracy", n_jobs = -1, random_state = 8)


test_read_and_validate()
