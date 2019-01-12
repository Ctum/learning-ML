import numpy as np
import operator

def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

def classify(inx, dataset, labels, k):
    # shape[0] 返回行数, [1]返回列数
    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    # axis =1 表示行相加 axis=0表示列相加
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # itemgetter(1) 根据值排序, 0 根据键排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, labels = createDataSet()
    test = [101, 20]
    # k 表示 k近邻
    test_class = classify(test, group, labels, 1)
    print(test_class)

