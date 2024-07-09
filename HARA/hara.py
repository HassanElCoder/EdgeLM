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
    C_params = np.logspace(-6, 3, 10)
    train_scores,val_scores=cross_validate_linear_svm(X_train.values,y_train.values.flatten(),param_name = "C",param_range = C_params, cv = 5, scoring = "accuracy", n_jobs = -1, random_state = 8)
    return train_scores,val_scores


def plot_accuracy(train_scores, val_scores, C_params):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # you can update these values based on the ploted scores for better view
    y_min = 0.5
    y_max = 1.0

    f = plt.figure(figsize = (12, 8))
    ax = plt.axes()
    plt.title("SVM Training and Validation Accuracy")
    plt.xlabel("C Value")
    plt.ylabel("Accuracy")
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(y_min, y_max + .01, .05))
    plt.semilogx(C_params, train_scores_mean, label = "Training Accuracy", color = "red")
    plt.fill_between(C_params, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha = 0.2, color = "red")
    plt.semilogx(C_params, val_scores_mean, label = "Validation Accuracy",
                 color = "green")
    plt.fill_between(C_params, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha = 0.2, color = "green")
    plt.legend(loc = "best")

    plt.show()



if __name__ == '__main__':
   train_scores,val_scores=test_read_and_validate()
   C_params = np.logspace(-6, 3, 10)
   plot_accuracy(train_scores, val_scores, C_params)

