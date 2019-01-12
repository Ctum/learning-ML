import numpy as np

from KNN.KNN_test01 import classify
from util.genderData import loadData
from util.trainTestSplit import train_test_split

dataSet = loadData()

train_set, train_label, test_set, test_label = train_test_split(dataSet)

total = test_set.shape[0]
error = 0
i_index = 0
for item in test_set:
    if not classify(item, train_set, train_label, 10) == test_label[i_index]:
        error = error + 1
    i_index = i_index + 1
print("error rate:", error / total)
