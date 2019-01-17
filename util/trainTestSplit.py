from numpy import *


def train_test_split(dataSet, ratio=0.8):
    """
    随机切分
    :param dataSet:
    :param ratio:
    :return train_set,train_label,test_set, test_label:
    """
    size = dataSet.shape[0]
    data_size = round(size * ratio)
    random.shuffle(dataSet)
    train = dataSet[:data_size,:]
    test = dataSet[:data_size, :]

    train_set = train[:, 0:3].copy()
    train_label = train[:, -1:].copy().flatten()

    test_set = test[:, 0:3].copy()
    test_label = test[:, -1:].copy().flatten()

    return train_set, train_label, test_set, test_label
