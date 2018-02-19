from sklearn.model_selection import train_test_split
import sys
import os
import os.path
import numpy as np

def labelsToDict(classLabelsStringsPath):
    labelDict = dict()
    counter = 0
    with open(classLabelsStringsPath) as f:
        labelStrings = f.readlines()
    for labelString in labelStrings:
        labelDict[labelString.rstrip()] = counter
        counter += 1
    return labelDict

def trainingPathToLabel(Xstring,labelDict):
    head, _ = os.path.split(Xstring)
    head, _ = os.path.split(head)
    _, classLabel = os.path.split(head)
    return labelDict[classLabel]

def getSplit(fileStringsPath,classLabelStringsPath,testSplit):
    with open(fileStringsPath,'r') as f:
        dataset = f.readlines()
    labelDict = labelsToDict(classLabelStringsPath)
    y = [trainingPathToLabel(X, labelDict) for X in dataset]
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(dataset,y,stratify=y,test_size = testSplit)
    yArray = np.array(Ytest)
    return Xtrain, Xtest, Ytrain, Ytest


def Main():
    if len(sys.argv) > 2 and \
        os.path.isfile(sys.argv[1]) and \
        os.path.isfile(sys.argv[2]):
        Xtrain, Xtest, Ytrain, Ytest = getSplit(sys.argv[1],sys.argv[2],0.1)
        with open('modelnet40_train.csv','w') as fTrain:
            fTrain.writelines(Xtrain)
        with open('modelnet40_val.csv','w') as fVal:
            fVal.writelines(Xtest)

    else:
        print('Files not found.')

Main()