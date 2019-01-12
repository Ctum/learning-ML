import os
import re
from numpy import *

def loadData():
    path = '../dataset/genderdata'
    files = os.listdir(path)
    trainList = []
    height = []
    for file in files:
        if not os.path.isdir(file):
            label=1
            if re.search("girl|female", file, re.I):
                label = 0
            with open(path + "\\" + file) as trainData:
                data = trainData.readlines()
                for item in data:
                    pattern = re.compile("^(\d+\.?\d*)\D+(\d+\.?\d*)\D+(\d+\.?\d*)$")
                    match = re.match(pattern, item)
                    if (match):
                        oneList = [float(match.group(1)), float(match.group(2)), float(match.group(3)), label]
                        trainList.append(oneList)
    return array(trainList)
