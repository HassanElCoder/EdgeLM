import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
random.seed(7)

data_path="./data/HAPTDataSet/"



def readdata(datapath):
    X_train = pd.read_table(os.path.join(datapath,'Train/X_train.txt'), header=None, sep=" ")
    y_train = pd.read_table(os.path.join(datapath,'Train/y_train.txt'), header = None, sep = " ")
    X_test = pd.read_table(os.path.join(datapath,'Test/X_test.txt'), header=None, sep=" ")
    y_test = pd.read_table(os.path.join(datapath,'Test/y_test.txt'), header=None, sep=" ")
    return X_train,y_train,X_test,y_test





