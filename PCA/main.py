'''
主成分分析 principal component analysis
算法步骤:
零均值化:每个特征的均值变成0
->
求协方差矩阵, numpy中cov(), 其中rowvar 很重要，rowvar=0,表示一行代表一个样本
->
求得特征值和特征向量,linalg中eig函数
-> 主成分个数选择
'''

import numpy as np
from util.genderData import loadData
from util.trainTestSplit import train_test_split

dataSet = loadDataSet()
train_set, train_label, test_set, test_label = train_test_split(dataSet)

def pca(data):
    rows, cols = data.shape
    data_mean = np.mean(data, 0)
    # 零值化
    Z = data - np.tile(data_mean, (rows, 1))
    D, V = np.linalg.eig(Z * Z.T)
    eigValidance = np.argsort(D)

def pac_action(dataMat, topNfeat=999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(covMat)
    eigValid = np.argsort(eigVals)
    eigValid = eigValid[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValid]
    lowDDataMat = meanRemoved * redEigVects
    return lowDDataMat

low = pac_action(train_set,1)
print(low.shape)