from numpy import *
import operator
import os
import re
import matplotlib.pyplot as plt
from util.genderData import loadData
from util.trainTestSplit import train_test_split

dataSet = loadData()
train_set,train_label,test_set, test_label = train_test_split(dataSet)


def param(feature):
    # 返回 p1,p2 均值 方差(协方差)
    rows, cols = feature.shape
    p1,p2
    if cols == 1:
        p1 = mean(feature, axis=0)[0]
        p2 = var(feature, axis=0)[0]
    else:
        p1 = mean(feature, axis=0)
        p2 = cov(feature)
    print(p1, p2)
    return p1, p2
