import os
import pdb
import argparse
import numpy as np
import sklearn.svm
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import re

NUM_AUGMENT = 10

def labelsToDict(classLabelsStringsPath):
    labelDict = dict()
    counter = 0
    with open(classLabelsStringsPath) as f:
        labelStrings = f.readlines()
    for labelString in labelStrings:
        labelDict[labelString.rstrip()] = counter
        counter += 1
    return labelDict

def getData(datasetName,featureModel,augment,finetune,fold=None):
    dataset = None
    if finetune:
        print('FineTuning!')
    if featureModel == 2:
        featureModelStr = 'model2'
    elif featureModel == 3:
        featureModelStr = 'model3'
    else:
        raise ValueError('Not a valid feature model!')

    if datasetName == 'modelnet10':
        with open('./preprocessing/modelnet10_trainvaltest.csv','r') as tvtFile:
            datasetFiles = tvtFile.readlines()
            datasetLabelStrings = [x.split('/')[-3] for x in datasetFiles]
            labelDict = labelsToDict('./preprocessing/modelnet10_labels.csv')
            labels = np.array([labelDict[x] for x in datasetLabelStrings])
    elif datasetName == 'modelnet40':
        with open('./preprocessing/modelnet40_auto_aligned_trainvaltest.csv','r') as tvtFile:
            datasetFiles = tvtFile.readlines()
            datasetLabelStrings = [x.split('/')[-3] for x in datasetFiles]
            labelDict = labelsToDict('./preprocessing/modelnet40_labels.csv')
            labels = np.array([labelDict[x] for x in datasetLabelStrings])
    elif datasetName == 'sydney':
        with open('./preprocessing/sydneyfoldsall.csv','r') as tvtFile:
            datasetFiles = tvtFile.readlines()
            datasetLabelStrings = [re.split('[/.]',x)[-4] for x in datasetFiles]
            labelDict = labelsToDict('./preprocessing/sydney_labels.csv')
            labels = np.array([labelDict[x] for x in datasetLabelStrings])
    elif datasetName == 'bosphorus':
        with open('./preprocessing/bosphorus_emotions.csv','r') as tvtFile:
            datasetFiles = tvtFile.readlines()
            datasetLabelStrings = [x.split('_')[-2] for x in datasetFiles]
            labelDict = labelsToDict('./preprocessing/bosphorus_labels.csv')
            labels = np.array([labelDict[x] for x in datasetLabelStrings])
    if finetune:
        fineTuneStr = 'finetune/'
    else:
        fineTuneStr = ''
    if fold is not None and finetune:
        foldStr = '_' + str(fold)
    else:
        foldStr = ''
    datasetStr = './preprocessing/features/' + datasetName + '/' + fineTuneStr + featureModelStr + '/features' + foldStr + '.npy'
    dataset = np.load(datasetStr)
    labelSet = labels
    if augment:
        if datasetName == 'modelnet10' or datasetName == 'modelnet40':

            for i in range(1,NUM_AUGMENT+1):
                augTrain = np.load('./preprocessing/features/' + datasetName + '/' + fineTuneStr + featureModelStr +
                                            '/features_train' + str(i) + '.npy')
                augVal = np.load('./preprocessing/features/' + datasetName + '/' + fineTuneStr + featureModelStr +
                                            '/features_val' + str(i) + '.npy')
                augTest = np.load('./preprocessing/features/' + datasetName + '/' + fineTuneStr + featureModelStr +
                                            '/features_test' + str(i) + '.npy')
                dataset = np.concatenate((dataset,augTrain,augVal,augTest))
                labelSet = np.concatenate((labelSet,labels))
        else:
            for i in range(1,NUM_AUGMENT+1):
                aug = np.load('./preprocessing/features/' + datasetName + '/' + fineTuneStr + featureModelStr +
                                            '/features_aug' + str(i) + foldStr + '.npy')
                dataset = np.concatenate((dataset,aug))
                labelSet = np.concatenate((labelSet,labels))

    return dataset, labelSet


def getIndices(datasetName,augment,isFinal=None,fold=None, kfolds=None):
    if datasetName == 'modelnet10':
        with open('./preprocessing/modelnet10_trainvaltest.csv','r') as tvtFile:
            datasetIndices = tvtFile.readlines()
        if isFinal:
            index = 0
            trainIdx = np.arange(index,index + 3991)
            index += 3991
            testIdx = np.arange(index,index+908)
            index += 908
            if augment:
                for i in range(NUM_AUGMENT):
                    trainIdx = np.concatenate((trainIdx,np.arange(index,index+3991)))
                    index += 3991
                    #testIdx = np.concatenate((testIdx,np.arange(index, index + 908)))
                    index += 908

        else:
            index = 0
            trainIdx = np.arange(index,index + 3589)
            index += 3589
            testIdx = np.arange(index,index+402)
            index += 402
            index += 908
            if augment:
                for i in range(NUM_AUGMENT):
                    trainIdx = np.concatenate((trainIdx,np.arange(index,index+3589)))
                    index += 3589
                    #testIdx = np.concatenate((testIdx,np.arange(index, index + 402)))
                    index += 402
                    index += 908

    elif datasetName == 'modelnet40':
        if isFinal:
            index = 0
            trainIdx = np.arange(index,index + 9843)
            index += 9843
            testIdx = np.arange(index,index+2468)
            index += 2468
            if augment:
                for i in range(NUM_AUGMENT):
                    trainIdx = np.concatenate((trainIdx,np.arange(index,index+9843)))
                    index += 9843
                    #testIdx = np.concatenate((testIdx,np.arange(index, index + 908)))
                    index += 2468
            #trainIdx = np.arange(9843)
            #testIdx = np.arange(9843,9843+2468)
        else:
            index = 0
            trainIdx = np.arange(index,index + 8858)
            index += 8858
            testIdx = np.arange(index,index+985)
            index += 985
            index += 2468
            if augment:
                for i in range(NUM_AUGMENT):
                    trainIdx = np.concatenate((trainIdx,np.arange(index,index+8858)))
                    index += 8858
                    #testIdx = np.concatenate((testIdx,np.arange(index, index + 985)))
                    index += 985
                    index += 2468
            #trainIdx = np.arange(8858)
            #testIdx = np.arange(8858,8858+985)
    elif datasetName == 'sydney':
        folds = [0, 1, 2, 3]
        foldStarts = [0,146,146+155,146+155+132]
        foldLengths= [146,155,132,155]
        trainFolds = [x for x in folds if x != fold]
        testIdx = np.arange(foldStarts[fold],foldStarts[fold] + foldLengths[fold])
        trainIdx = []
        for trainFold in trainFolds:
            trainIdx.append(np.arange(foldStarts[trainFold],foldStarts[trainFold] + foldLengths[trainFold]))
        trainIdx = np.concatenate(trainIdx)
        if augment:
            for i in range(1,NUM_AUGMENT+1):
                offset = i * 588
                foldStarts = [offset,offset + 146,offset + 146+155,offset + 146+155+132]
                foldLengths= [146,155,132,155]
                trainFolds = [x for x in folds if x != fold]
                testIdx = np.arange(foldStarts[fold],foldStarts[fold] + foldLengths[fold])
                trainIdxAug = []
                for trainFold in trainFolds:
                    trainIdxAug.append(np.arange(foldStarts[trainFold],foldStarts[trainFold] + foldLengths[trainFold]))
                trainIdxAug = np.concatenate(trainIdxAug)
                trainIdx = np.concatenate((trainIdx,trainIdxAug))
    elif datasetName == 'bosphorus':
        trainIdx, testIdx = kfolds[fold]

    print(trainIdx.shape)
    print(testIdx.shape)
    return (trainIdx, testIdx)

parser = argparse.ArgumentParser(description='Process input architecture')
parser.add_argument('--arch', default='1024_512_256', help='Defines the model')
parser.add_argument('--date', default='Nov25', help='Data run model')
parser.add_argument('--dataset_name', default='modelnet10', help='Dataset name')
#Add loading pretrained weights option
parser.add_argument('--loading_weights_flag', default=0, type=int,help='loading weights flag')

parser.add_argument('--path_pretrained_weights', default='/home/thn2079/git/GraphCNN_WACV/GraphCNN/Graph-CNN/snapshots/Modelnet10-Oct6/Modelnet10-OC-c_16_1_1-c_16_1_1-c_16_1_1-c_16_1_1-p_16-fc_10_0_0-l2=0.0-l1=0.0/model-7', help='Path to pretrained weights')
parser.add_argument('--train_flag', default=1, type=int,help='training flag')
parser.add_argument('--num_classes', default=10, type=int,help='Number of classes')
parser.add_argument('--is_final', default=0, type=int,help='Train with test set (final) or validation (not)')
parser.add_argument('--feature_model', default=2, type=int,help='What model should features come from (2 or 3)')
parser.add_argument('--C', default=1.0, type=float,help='Error penalty hyperparameter')
parser.add_argument('--kernel', default='rbf', help='Kernel Function (linear, poly, rbf, sigmoid)')
parser.add_argument('--poly_degree', default=3, type=int, help='Degree for polynomial kernel function')
parser.add_argument('--gamma', default=0.01, type=float, help='Gamma hyperparameter')
parser.add_argument('--coef0', default=0.0, type=float, help='Bias term in poly/sigmoid kernel functions')
parser.add_argument('--num_iter', default=100, type=int,help='Number of iterations')
parser.add_argument('--augment', default=0, type=int, help='Use augmented training data')
parser.add_argument('--tol', default=0.001, type=float, help='Tolerance for early stop')
parser.add_argument('--finetune', default=0, type=int, help='Use finetuned data')

args = parser.parse_args()



hidden_layers = [int(x) for x in args.arch.split('_')]

if args.dataset_name == 'modelnet10' or args.dataset_name == 'modelnet40':
    mode = 'fixed'
elif args.dataset_name == 'sydney':
    mode = 'kfold'
    K = 4
elif args.dataset_name == 'bosphorus':
    mode = 'kfold'
    K = 5
scaler = sklearn.preprocessing.StandardScaler()
if mode == 'fixed':
    dataset,labels = getData(args.dataset_name,args.feature_model,args.augment,args.finetune)
    print(dataset.shape)
    print(labels.shape)
    (trainIdx, testIdx) = getIndices(args.dataset_name,args.augment,args.is_final)
    svmClassifier = sklearn.svm.SVC(C=args.C, kernel=args.kernel, degree=args.poly_degree, gamma='auto', coef0=args.coef0,\
                                    shrinking=True,probability=False, tol=args.tol, cache_size=200, class_weight=None, \
                                    verbose=True, max_iter=args.num_iter, decision_function_shape='ovr', random_state=None)

    trainData = scaler.fit_transform(dataset[trainIdx,:].astype(np.float64))
    testData = scaler.fit_transform(dataset[testIdx,:].astype(np.float64))
    svmClassifier = svmClassifier.fit(trainData,labels[trainIdx])
    print('Training Finished')
    trainAcc = svmClassifier.score(trainData,labels[trainIdx])
    print('Final Accuracy on Training Set: {0}%'.format(trainAcc))
    currentResult = svmClassifier.score(testData,labels[testIdx])
    print('Final Accuracy on Validation Set: {0}%'.format(currentResult))
    trainPred = svmClassifier.predict(trainData)
    print('F1 Score on Training Set: {0}'.format(sklearn.metrics.f1_score(labels[trainIdx],trainPred,average='weighted')))
    testPred = svmClassifier.predict(testData)
    print('F1 Score on Validation Set: {0}'.format(sklearn.metrics.f1_score(labels[testIdx],testPred,average='weighted')))

elif mode == 'kfold':
    valResults = []
    trainResults = []
    valF1 = []
    trainF1 = []
    #if args.dataset_name == 'bosphorus':
     #   kfoldObj = sklearn.model_selection.StratifiedKFold(n_splits=K)
    #    kfolds = list(kfoldObj.split(dataset,labels))
    for i in range(K):
        dataset,labels = getData(args.dataset_name,args.feature_model,args.augment,args.finetune,i)
        print(dataset.shape)
        print(labels.shape)
        if args.dataset_name == 'sydney':
            (trainIdx, testIdx) = getIndices(args.dataset_name,args.augment,fold=i)
        elif args.dataset_name == 'bosphorus':
            (trainIdx, testIdx) = getIndices(args.dataset_name,args.augment,fold=i,kfolds=kfolds)
        svmClassifier = sklearn.svm.SVC(C=args.C, kernel=args.kernel, degree=args.poly_degree, gamma='auto', coef0=args.coef0,\
                                    shrinking=True,probability=False, tol=0.001, cache_size=200, class_weight=None, \
                                    verbose=True, max_iter=args.num_iter, decision_function_shape='ovr', random_state=None)

        trainData = scaler.fit_transform(dataset[trainIdx,:].astype(np.float64))
        testData = scaler.fit_transform(dataset[testIdx,:].astype(np.float64))
        svmClassifier = svmClassifier.fit(trainData,labels[trainIdx])
        trainAcc = svmClassifier.score(trainData,labels[trainIdx])
        print('Final Accuracy on Training Set: {0}%'.format(trainAcc))
        currentResult = svmClassifier.score(testData,labels[testIdx])
        print('Final Accuracy on Validation Set: {0}%'.format(currentResult))
        trainPred = svmClassifier.predict(trainData)
        testPred = svmClassifier.predict(testData)
        valResults.append(currentResult)
        trainResults.append(trainAcc)
        trainF1Current = sklearn.metrics.f1_score(labels[trainIdx],trainPred,average='weighted')
        trainF1.append(trainF1Current)
        print('F1 Score on Training Set: {0}'.format(trainF1))
        valF1Current = sklearn.metrics.f1_score(labels[testIdx],testPred,average='weighted')
        valF1.append(valF1Current)
        print('F1 Score on Validation Set: {0}'.format(valF1))

    print('Mean Train Accuracy Across Folds: {0}'.format(np.mean(np.array(trainResults))))
    print('Mean Validation Accuracy Across Folds: {0}'.format(np.mean(np.array(valResults))))
    print('Mean Train F1 Across Folds: {0}'.format(np.mean(np.array(trainF1))))
    print('Mean Validation F1 Across Folds: {0}'.format(np.mean(np.array(valF1))))
