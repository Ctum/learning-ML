from numpy import *
import operator
import os
import re
import matplotlib.pyplot as plt


class bayes_classifier_guassion:
    def __init__(self, male, female):
        self.male = male
        self.female = female

    def param(self, data):
        """
        返回 期望、方差、协方差矩阵
        :param data:
        :return:
        """
        if data.shape[1]==1:
            # 单变量
            data = data.flatten()
            a = sum(data) / data.shape[0]
            total = 0
            for i in data:
                total += (i-a) ** 2
            b = total / data.shape[0]
            c=0
            return a, b, c
        else:
            pass


if __name__ == '__main__':
    dataSet = loadDataSet()
    train, test = train_test_split(dataSet)
    print(train, 'train')
    print(test, 'test')
