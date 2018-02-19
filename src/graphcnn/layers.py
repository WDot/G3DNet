from graphcnn.helper import *
import tensorflow as tf
import numpy as np
import math
import os
import os.path
from tensorflow.contrib.layers.python.layers import utils
import pdb         

def _histogram_summaries(var, name=None):
    if name is None:
        return tf.summary.histogram(var.name, var)
    else:
        return tf.summary.histogram(name, var)
    
def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
    _histogram_summaries(var)
    return var
    
def make_bias_variable(name, shape):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
    _histogram_summaries(var)
    return var

def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.005):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), regularizer=regularizer)
    return var
    
def make_bn(input, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
    default_decay = GraphCNNGlobal.BN_DECAY
    with tf.variable_scope(name, default_name='BatchNorm') as scope:
        input_size = input.get_shape()[axis].value
        if axis == -1:
            axis = len(input.get_shape())-1
        axis_arr = [i for i in range(len(input.get_shape())) if i != axis]
        if mask == None:
            batch_mean, batch_var = tf.nn.moments(input, axis_arr)
        else:
            batch_mean, batch_var = tf.nn.weighted_moments(input, axis_arr, mask)
        gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
        beta = make_bias_variable('bias', input_size)
        ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)
        
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        batch_norm = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
        _histogram_summaries(batch_norm, 'batch_norm')
        return batch_norm

      
def batch_mat_mult(A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    
    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
        axis_2 = -1
    else:
        axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
    return result
    
def make_softmax_layer(V, axis=1, name=None):
    with tf.variable_scope(name, default_name='Softmax') as scope:
        max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
        exp = tf.exp(tf.subtract(V, max_value))
        prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
        _histogram_summaries(prob)
        return prob
    
def make_graphcnn_layer(V, A, no_filters, stride=1, order=1, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        weightList = []
        no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])
        result = batch_mat_mult(V, W_I) + b
        weightList.append(W_I)
        
        Acurrent = A
        for k in range(1,order + 1):
            if k % stride == 0:
                with tf.variable_scope('Order' + str(k)) as scope:
                    W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
                    A_shape = tf.shape(Acurrent)
                    A_reshape = tf.reshape(Acurrent, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
                    n = tf.matmul(A_reshape, V)
                    n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])
                    result = batch_mat_mult(n, W)
                    weightList.append(W)
            Acurrent = tf.transpose(tf.matmul(tf.transpose(Acurrent,[0,2,1,3]), tf.transpose(A,[0,2,1,3])),[0,2,1,3])
        _histogram_summaries(Acurrent, "Acurrent")
        _histogram_summaries(result, "Result")
        return result, weightList

def make_sparse_graphcnn_layer(V, A, no_filters, stride=1, order=1, name=None):
    here = os.path.dirname(__file__) + '/util/ops/'
    #For now, assume no weights. This just tests A*V
    if os.path.isfile(os.path.join(here, 'SparseConv.so')):
        _graphcnn_conv_sparse_module = tf.load_op_library(os.path.join(here, 'SparseConv.so'))
    if isinstance(A, tf.Tensor):
        no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value

        Acurrent = A
        for k in range(1,order + 1):
            if k % stride == 0:
                with tf.variable_scope('Order' + str(k)) as scope:
                    #W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
                    A_shape = tf.shape(Acurrent)
                    A_reshape = tf.reshape(Acurrent, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
                    n = tf.matmul(A_reshape, V)
                    n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])
                    result = tf.reduce_sum(n,axis=2)
            Acurrent = tf.transpose(tf.matmul(tf.transpose(Acurrent,[0,2,1,3]), tf.transpose(A,[0,2,1,3])),[0,2,1,3])
        _histogram_summaries(Acurrent, "Acurrent")
        _histogram_summaries(result, "Result")
        return result, []
    elif isinstance(A, tf.SparseTensorValue):
        return _graphcnn_conv_sparse_module.sparse_graph_convolution(V, A.indices, A.values, num_filters=no_filters), []


def make_graph_embed_pooling(V, A, no_vertices=1, mask=None, name=None):
    with tf.variable_scope(name, default_name='GraphEmbedPooling') as scope:
        factors, W = make_embedding_layer(V, no_vertices, name='Factors')
        
        if mask is not None:
            factors = tf.multiply(factors, mask)  
        factors = make_softmax_layer(factors)
        
        result = tf.matmul(factors, V, transpose_a=True)
        
        if no_vertices == 1:
            no_features = V.get_shape()[2].value
            return tf.reshape(result, [-1, no_features]), A
        
        result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
        result_A = tf.matmul(factors, result_A, transpose_a=True)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], no_vertices, A.get_shape()[2].value, no_vertices))
        _histogram_summaries(result, "result")
        _histogram_summaries(result_A, "result_a")
        return result, result_A, W
    
def make_embedding_layer(V, no_filters, name=None):
    with tf.variable_scope(name, default_name='Embed') as scope:
        no_features = V.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0/math.sqrt(no_features))
        b = make_bias_variable('bias', [no_filters])
        V_reshape = tf.reshape(V, (-1, no_features))
        s = tf.slice(tf.shape(V), [0], [len(V.get_shape())-1])
        s = tf.concat([s, tf.stack([no_filters])], 0)
        result = tf.reshape(tf.matmul(V_reshape, W) + b, s)
        _histogram_summaries(result, "result")
        return result, W

#MASK should be LxNxN, even if L = 1
#Will of course need to reshape later
def make_mask_block(Aaccprev, Akprev, A, name=None):
    with tf.name_scope(name, default_name='MaskBlock') as scope:
        Aacc = tf.minimum(Aaccprev + Akprev, tf.ones(tf.shape(Aaccprev)))
        # pdb.set_trace()
        mm = tf.transpose(tf.matmul(tf.transpose(Akprev,[0,2,1,3]), tf.transpose(A,[0,2,1,3])),[0,2,1,3])
        Ak = tf.minimum(mm, tf.ones(tf.shape(Akprev)))
        Amask = Ak - Aacc
        # Aout = A
        
        _histogram_summaries(Amask, "Amask")    
    return Amask, Aacc, Ak, A

def make_init_mask_block(A, name=None):
    with tf.name_scope(name, default_name='InitMaskBlock') as scope:
        Ashape = tf.shape(A)
        no_A = Ashape[2]
        I = tf.eye(Ashape[1],batch_shape=[Ashape[0]])
        I = tf.transpose(tf.tile(tf.expand_dims(I,0), tf.stack([no_A,1,1,1])), [1,2,0,3])
        Aacc = tf.minimum(I + A, tf.ones(tf.shape(A)))
        Ak = A
        # Aout = Aacc
        # pdb.set_trace()
        Amask = tf.maximum(A - I, tf.zeros(tf.shape(A)))
        _histogram_summaries(Amask, "init_Amask")
    return Amask, Aacc, Ak, Aacc

def make_graphcnn_unbiased_layer(V, A, no_filters, Amask, Aaccprev, Akprev, Aout, stride=1, order=1, name=None, prev_layer=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        weightList = []
        if 'conv1' in scope.name:
            Amask, Aaccprev, Akprev, Aout = make_init_mask_block(A)
            A = Aout

        Amasked = tf.multiply(A,Amask)
        no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])
        Vout = batch_mat_mult(V, W_I) + b
        weightList.append(W_I)
        for k in range(1,order + 1):
            if k % stride == 0:
                with tf.variable_scope('Order' + str(k)) as scope:
                    W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
                    A_shape = tf.shape(Amasked)
                    A_reshape = tf.reshape(Amasked, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
                    n = tf.matmul(A_reshape, V)
                    n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])
                    Vout = Vout + batch_mat_mult(n, W)
                    weightList.append(W)
            Amask, Aaccprev, Akprev, Aout = make_mask_block(Aaccprev, Akprev, A)
            Amasked = tf.multiply(Akprev,Amask)
        # pdb.set_trace()        
        _histogram_summaries(Amasked, "Amasked")
        return Vout,  Amask, Aaccprev, Akprev, Aout, weightList
        
def make_graph_pooling_layer(V, A, P, name=None):
    with tf.variable_scope(name,default_name='Graph-Pooling') as scope:
        Vout = tf.matmul(tf.transpose(P,perm=[0,2,1]),V)
        Ashape = tf.shape(A)
        Prep = tf.tile(tf.expand_dims(P,2),[1,1,Ashape[2],1])
        Ptranspose = tf.transpose(Prep,perm=[0,2,3,1])
        Pnottranspose = tf.transpose(Prep,perm=[0,2,1,3])
        Abatched = tf.transpose(A,perm=[0,2,1,3])
        leftMultiply = tf.matmul(Ptranspose,Abatched)
        rightMultiply = tf.matmul(leftMultiply,Pnottranspose)
        Aout = tf.transpose(rightMultiply,perm=[0,2,1,3])
        return Vout, Aout

def make_graph_maxpooling_layer(V, A, P, name=None):
    with tf.variable_scope(name,default_name='Graph-Pooling') as scope:
        Pextend = tf.expand_dims(tf.transpose(P,perm=[0,2,1]),3)
        Vextend = tf.expand_dims(V,1)
        #Use broadcasting tricks to get the maximum vertex of each cluster
        #Each column of P^T is an indicator of whether that vertex is a candidate
        #in a given coarse cluster
        #The number of rows is the number of coarse vertices
        #We want to mutiply each individual vertex feature vector by the scalar indicator
        #Then take the maximum for each coarse vertex
        Vout = tf.reduce_max(tf.multiply(Pextend,Vextend),axis=2)
        #Vout = tf.matmul(tf.transpose(P,perm=[0,2,1]),V)
        Ashape = tf.shape(A)
        Prep = tf.tile(tf.expand_dims(P,2),[1,1,Ashape[2],1])
        Ptranspose = tf.transpose(Prep,perm=[0,2,3,1])
        Pnottranspose = tf.transpose(Prep,perm=[0,2,1,3])
        Abatched = tf.transpose(A,perm=[0,2,1,3])
        leftMultiply = tf.matmul(Ptranspose,Abatched)
        rightMultiply = tf.matmul(leftMultiply,Pnottranspose)
        Aout = tf.transpose(rightMultiply,perm=[0,2,1,3])
        return Vout, Aout


def make_graph_unpooling_layer(V, A, P, name=None):
    with tf.variable_scope(name, default_name='Graph-Unpooling') as scope:
        Vout = tf.matmul(P, V)
        Ashape = tf.shape(A)
        Prep = tf.tile(tf.expand_dims(P, 2), [1, 1, Ashape[2], 1])
        Ptranspose = tf.transpose(Prep, perm=[0, 2, 3, 1])
        Pnottranspose = tf.transpose(Prep, perm=[0, 2, 1, 3])
        Abatched = tf.transpose(A, perm=[0, 2, 1, 3])
        leftMultiply = tf.matmul(Pnottranspose, Abatched)
        rightMultiply = tf.matmul(leftMultiply, Ptranspose)
        Aout = tf.transpose(rightMultiply, perm=[0, 2, 1, 3])
        return Vout, Aout

        
        
