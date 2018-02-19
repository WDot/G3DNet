from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
# from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import pcl
import time
from graphcnn.experiments.experiment import GraphCNNExperiment
from tensorflow.python.training import queue_runner
import pickle as pkl
from collections import defaultdict
import transforms3d
import numpy.random as random
import math
from graphcnn.util.pooling.GeometricAdjacencyCompander import GeometricAdjacencyCompander
from graphcnn.util.pooling.PoolingFactory import PoolingFactory
import scipy.spatial
from PIL import Image

# This function is used to create tf.cond compatible tf.train.batch alternative
def _make_batch_queue(input, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input],shapes=[s.get_shape() for s in input])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
           tf.cast(queue.size(), tf.float32) *
           (1. / capacity))
    enqueue_ops = [queue.enqueue(input)]*num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue


# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNPCDExperiment(GraphCNNExperiment):
    def __init__(self, dataset_name, model_name, net_constructor, prefix, N, C, L, K, numClasses, trainFile, testFile,\
                 label_file, isTrain,l2=0,l1=0,feature_type=1,poolRatios=None,path_pretrained_weights=None,\
                 poolingId='Lloyd'):
        GraphCNNExperiment.__init__(self, dataset_name, model_name, net_constructor,l2,l1,path_pretrained_weights)
        self.prefix = prefix
        self.N = N
        self.C = C
        self.L = L
        self.K = K
        self.numClasses = numClasses
        self.trainFile = trainFile
        self.testFile = testFile
        self.isTrain = isTrain
        self.feature_type = feature_type
        with open(label_file, 'r') as f:
            labels = f.read().splitlines()
        self.labels = dict(zip(labels, range(len(labels))))
        self.poolRatios = poolRatios
        self.poolingId = poolingId
        if self.poolRatios is not None:
            self.poolFactory = PoolingFactory()

    # Create input_producers and batch queues
    def read_file_path(self, file):
        labels = []
        with open(file, 'r') as f:
            list_files = pkl.load(f)
        return map(lambda x: tf.convert_to_tensor(x), list_files)

    def calculate_features_wrap_train(self, file):
        return tf.py_func(self.calculate_features_train, [file], [tf.float32, tf.float32, tf.int64])
        
    def calculate_features_wrap_test(self, file):
        return tf.py_func(self.calculate_features_test, [file], [tf.float32, tf.float32, tf.int64])
    
    def get_single_sample(self, trainExample,isTrain):

        if isTrain:
            trainV, trainA, trainLabel = self.calculate_features_wrap_train(trainExample)
        else:
            trainV, trainA, trainLabel = self.calculate_features_wrap_test(trainExample)
        trainV = tf.reshape(trainV, [self.N, self.C])
        trainA = tf.reshape(trainA, [self.N, self.L, self.N])
        trainLabel = tf.one_hot(trainLabel,self.numClasses)
        trainLabel.set_shape([self.numClasses])
        single_sample = [trainV, trainA, trainLabel]
        if self.poolRatios is not None:
            pooler = self.poolFactory.CreatePoolingPyramid(len(self.poolRatios), GeometricAdjacencyCompander,\
                                                       self.poolRatios,self.poolingId)
            Plist = tf.py_func(pooler.makeP,[trainA,trainV],[tf.float32]*len(self.poolRatios),stateful=False)
            prevSize = self.N
            for P in Plist:
                currentSize = np.floor(prevSize * 0.5)
                P.set_shape([prevSize, currentSize])
                prevSize = currentSize
            single_sample += Plist
        
        return single_sample

    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')
                    trainQueue_file = tf.train.string_input_producer([self.trainFile], shuffle=True, seed=1000)
                    reader = tf.TextLineReader()
                    _, trainExample = reader.read(trainQueue_file)

                    single_sample_train = self.get_single_sample(trainExample,True)
                    train_queue = _make_batch_queue(single_sample_train, capacity=self.train_batch_size * 2, num_threads=1)
                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')

                    testQueue_file = tf.train.string_input_producer([self.testFile])
                    reader = tf.TextLineReader()
                    _, testExample = reader.read(testQueue_file)

                    single_sample_test = self.get_single_sample(testExample,False)
                    test_queue = _make_batch_queue(single_sample_test, capacity=self.test_batch_size * 2, num_threads=1)
                return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size) )

    def create_data_test(self):
        with tf.device("/cpu:0"):
             with tf.variable_scope('test_data') as scope:
                self.print_ext('Creating test Tensorflow Tensors')

                testQueue_file = tf.train.string_input_producer([self.testFile], num_epochs=1, shuffle=False)
                reader = tf.TextLineReader()
                _, testExample = reader.read(testQueue_file)

                single_sample_test = self.get_single_sample(testExample,False)
                test_queue = _make_batch_queue(single_sample_test, capacity=self.test_batch_size * 2, num_threads=1)
                # return test_queue.dequeue_many(self.test_batch_size)
                # return test_queue.dequeue_up_to(self.test_batch_size)
                return tf.train.batch(test_queue.dequeue(), self.test_batch_size, num_threads=1,dynamic_pad=True, allow_smaller_final_batch=True)

    def calculate_features_test(self,input):
        return self.calculate_features(input,self.K,self.N,False,self.feature_type)

    def calculate_features_train(self,input):
        return self.calculate_features(input,self.K,self.N,True,self.feature_type)
                
    def calculate_features(self, input, K=3, MAX_SIZE=500, splitNeighbors=True, aug=False,feature_type=1):##input tensor = NxNx3

        inputData = self.prefix + '/' + input
        label = self.labels[input.split('/')[-3]]
        cloud = pcl.load(inputData)
        #stolen from ECC code, drops out random points
        #if aug:
            #Probability a point is dropped
        p = 0.1
        cloudArray = cloud.to_array()
        keptIndices = random.choice(range(cloudArray.shape[0]), size=int(math.ceil((1-p)*cloudArray.shape[0])),replace=False)
        cloudArray = cloudArray[keptIndices,:]
        cloud.from_array(cloudArray)
        cloud.resize(MAX_SIZE)

        xyz = cloud.to_array()[:,:3]  

        #Stolen from ECC code
        if aug:
            M = np.eye(3)
            s = random.uniform(1/1.1, 1.1)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)            
            #angle = random.uniform(0, 2*math.pi)
            #M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption        
            if random.random() < 0.5/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
            if random.random() < 0.5/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
            xyz = np.dot(xyz,M.T)
        if feature_type == 1:
            kd = pcl.KdTreeFLANN(cloud)
            #if aug:
            #    currentK = random.randint(np.maximum(1,K-2),K+2)
            #    indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, currentK)
            #else:
            indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, K) # K = 2 gives itself and other point from cloud which is closest

            vertexMean = np.mean(xyz, axis=0)
            vertexStd = np.std(xyz, axis=0)
            #Jiggle the model a little bit if it is perfectly aligned with the axes
            #print(input)
            if not vertexStd.all():
                M = np.eye(3)
                angle = random.uniform(0.01,0.1,size=3)
                sign = random.choice([-1,1],size=3,replace=True)
                M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], sign[0] * angle[0]), M) 
                M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], sign[1] * angle[1]), M) 
                M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], sign[2] * angle[2]), M)
                xyz = np.dot(xyz,M.T)
                vertexMean = np.mean(xyz, axis=0)
                vertexStd = np.std(xyz, axis=0)
            xyz = (xyz - vertexMean)/vertexStd 

            num_nodes = xyz.shape[0]

            sqr_distances[:,0] += 1 #includes self-loops
            valid = np.logical_or(indices > 0, sqr_distances>1e-10)
            rowi, coli = np.nonzero(valid)
            idx = indices[(rowi,coli)]
            
            #print("XYZ Shape {0}".format(xyz.shape))
            edges = np.vstack([idx, rowi]).transpose()
            #print("Edge Shape {0}".format(edges.shape))
            A = np.zeros(shape=(MAX_SIZE,8, MAX_SIZE))
            zindices = np.dot([4, 2, 1], np.greater((xyz[edges[:,0],:] - xyz[edges[:,1],:]).transpose(), np.zeros((3,edges.shape[0]))));
            edgeLen = 1
            # print('From {0} to {1}: Len {2}',i,j,edgeLen)
            A[edges[:,0], zindices, edges[:,1]] = edgeLen
            A[edges[:,1], zindices, edges[:,0]] = edgeLen

        elif feature_type == 0:
            RADIUS = 0.5
            vertexMean = np.mean(xyz, axis=0)
            vertexStd = np.std(xyz, axis=0)
            #Jiggle the model a little bit if it is perfectly aligned with the axes
            #print(input)
            if not vertexStd.all():
                M = np.eye(3)
                angle = np.random.uniform(0.01,0.1,size=3)
                sign = np.random.choice([-1,1],size=3,replace=True)
                M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], sign[0] * angle[0]), M)
                M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], sign[1] * angle[1]), M)
                M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], sign[2] * angle[2]), M)
                V = np.dot(V,M.T)
                vertexMean = np.mean(V, axis=0)
                vertexStd = np.std(V, axis=0)
            V = (V - vertexMean)/vertexStd
            #V = np.pad(V,pad_width=((0,MAX_SIZE - V.shape[0]),(0,0)),mode='constant')
            kdtree = scipy.spatial.KDTree(V)
            knns = kdtree.query_ball_tree(kdtree,r=RADIUS)
            A = np.zeros(shape=(MAX_SIZE,8, MAX_SIZE))
            numNeighbors = [len(x) for x in knns]
            v1 = np.repeat(np.arange(V.shape[0]),numNeighbors)
            knnsStack = np.concatenate(knns)
            zindex = np.dot([4, 2, 1], np.greater((V[v1] - V[knnsStack]).transpose(), np.zeros((3,len(knnsStack)))));
            edgeLen = 1
            A[v1, zindex, knnsStack] = edgeLen
            A[knnsStack,zindex, v1] = edgeLen

        return xyz.astype(np.float32), A.astype(np.float32), label
