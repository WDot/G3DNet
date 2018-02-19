from graphcnn.layers import *
from graphcnn.network_description import GraphCNNNetworkDescription
import tensorflow as tf
import pdb
class GraphCNNNetwork(object):
    def __init__(self):
        self.current_V = None
        self.current_A = None
        self.weightList = []
        self.current_mask = None
        self.labels = None
        self.network_debug = False
        
        with tf.device("/cpu:0"):
            self.Amask = None
            self.Aaccprev = None
            self.Akprev = None
            self.Aout = None
        
        
        
    def create_network(self, input):
        self.current_V = input[0]
        self.current_A = input[1]
        self.labels = input[2]
        #if len(input) > 3:
        #    self.current_mask = input[3]
        #else:
        #    self.current_mask = None

        if len(input) > 3:
            self.current_Ps = input[3:]
        else:
            self.current_Ps = None
        
        if self.network_debug:
            if self.current_mask:
                size = tf.reduce_sum(self.current_mask, axis=1)
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), tf.reduce_max(size), tf.reduce_mean(size)], message='Input V Shape, Max size, Avg. Size:')
            else:
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V)], message='Input V Shape')
        return input
        
        
    def make_batchnorm_layer(self):
        self.current_V = make_bn(self.current_V, self.is_training, mask=self.current_mask, num_updates = self.global_step)
        return self.current_V
        
    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed') as scope:
            self.current_V, W = make_embedding_layer(self.current_V, no_filters)
            self.weightList.append(W)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V, self.current_A, self.current_mask
        
    def make_dropout_layer(self, keep_prob=0.5):
        self.current_V = tf.cond(self.is_training, lambda:tf.nn.dropout(self.current_V, keep_prob=keep_prob), lambda:(self.current_V))
        return self.current_V
        
    def make_graphcnn_layer(self, no_filters, stride=1, order=1, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.current_V, weightList = make_graphcnn_layer(self.current_V, self.current_A, no_filters, stride, order)
            self.weightList += weightList
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V
        
    def make_graphcnn_resnet_layer(self, no_filters, stride=1, order=1, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN-resnet') as scope:
            V_origin = tf.identity(self.current_V)
            self.make_graphcnn_layer(no_filters, stride, order,name+'-conv1')
            self.make_graphcnn_layer(no_filters, stride, order,name+'-conv2')
            self.make_graphcnn_layer(no_filters, stride, order,name+'-conv3')
            self.current_V+=V_origin
        return self.current_V

    def make_graphcnn_resnet_or_densenet_layer(self, no_filters=[16,16,16], stride=[1,1,1], order=[1,1,1], name=None, with_bn=[True,True, True], with_act_func=[True, True, True],type=0):
        ###type==0: Resnet skip connection block
        ###type==1: Densenet block, all are connected to each other
        if type==0:
            with tf.variable_scope(name, default_name='Graph-CNN-resnet') as scope:
                V_origin = tf.identity(self.current_V)
                for i in xrange(len(no_filters)):
                    self.make_graphcnn_layer(no_filters[i], stride[i], order[i],name+'-conv'+str(i))
                self.current_V+=V_origin
        elif type==1:
            with tf.variable_scope(name, default_name='Graph-CNN-resnet') as scope:
                # V_origin = tf.identity(self.current_V)
                list_V = []
                for i in xrange(len(no_filters)):
                    list_V.append(tf.identity(self.current_V))
                    self.make_graphcnn_layer(no_filters[i], stride[i], order[i],name+'-conv'+str(i))
                    for j in xrange(len(list_V)):
                        self.current_V+=list_V[j]
        return self.current_V
        
    def make_graphcnn_unbiased_resnet_or_densenet_layer(self, no_filters=[16,16,16], stride=[1,1,1], order=[1,1,1], name=None, with_bn=[True,True, True], with_act_func=[True, True, True],type=0):
        ###type==0: Resnet skip connection block
        ###type==1: Densenet block, all are connected to each other
        if type==0:
            with tf.variable_scope(name, default_name='Graph-CNN-resnet') as scope:
                V_origin = tf.identity(self.current_V)
                for i in xrange(len(no_filters)):
                    self.make_graphcnn_unbiased_layer(no_filters[i], stride[i], order[i],name+'-conv'+str(i))
                self.current_V+=V_origin
        elif type==1:
            with tf.variable_scope(name, default_name='Graph-CNN-resnet') as scope:
                # V_origin = tf.identity(self.current_V)
                list_V = []
                for i in xrange(len(no_filters)):
                    list_V.append(tf.identity(self.current_V))
                    self.make_graphcnn_unbiased_layer(no_filters[i], stride[i], order[i],name+'-conv'+str(i))
                    for j in xrange(len(list_V)):
                        self.current_V+=list_V[j]
        return self.current_V
      
    def make_graphcnn_unbiased_layer(self, no_filters, stride=1, order=1, name=None, with_bn=True, with_act_func=True, prev_layer=None):
        # pdb.set_trace()
        with tf.variable_scope(name, default_name='Graph-CNN-biased') as scope:
            self.current_V, self.Amask, self.Aaccprev, self.Akprev, self.Aout, weightList = make_graphcnn_unbiased_layer(self.current_V, self.current_A, no_filters, self.Amask, self.Aaccprev, self.Akprev, self.Aout, stride, order, name=None, prev_layer=prev_layer)
            self.weightList += weightList
            self.current_A = self.Akprev
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V, self.Amask, self.Aaccprev, self.Akprev, self.Aout
    
    def make_graph_embed_pooling(self, no_vertices=1, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='GraphEmbedPool') as scope:
            self.current_V, self.current_A, W = make_graph_embed_pooling(self.current_V, self.current_A, mask=self.current_mask, no_vertices=no_vertices)
            self.current_mask = None
            self.weightList.append(W)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='Pool "%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V, self.current_A, self.current_mask
            
    def make_fc_layer(self, no_filters, name=None, with_bn=False, with_act_func=True):
        with tf.variable_scope(name, default_name='FC') as scope:
            self.current_mask = None
            
            if len(self.current_V.get_shape()) > 2:
                no_input_features = int(np.prod(self.current_V.get_shape()[1:]))
                self.current_V = tf.reshape(self.current_V, [-1, no_input_features])
            self.current_V, W = make_embedding_layer(self.current_V, no_filters)
            self.weightList.append(W)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V
        
        
    def make_cnn_layer(self, no_filters, name=None, with_bn=False, with_act_func=True, filter_size=3, stride=1, padding='SAME'):
        with tf.variable_scope(None, default_name='conv') as scope:
            dim = self.current_V.get_shape()[-1]
            kernel = make_variable_with_weight_decay('weights',
                                                 shape=[filter_size, filter_size, dim, no_filters],
                                                 stddev=math.sqrt(1.0/(no_filters*filter_size*filter_size)),
                                                 wd=0.0005)
            conv = tf.nn.conv2d(self.current_V, kernel, [1, stride, stride, 1], padding=padding)
            biases = make_bias_variable('biases', [no_filters])
            self.current_V = tf.nn.bias_add(conv, biases)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            return self.current_V
            
    def make_pool_layer(self, padding='SAME'):
        with tf.variable_scope(None, default_name='pool') as scope:
            dim = self.current_V.get_shape()[-1]
            self.current_V = tf.nn.max_pool(self.current_V, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding, name=scope.name)

            return self.current_V

    def make_graph_pooling_layer(self, Pindex, name=None):
        with tf.variable_scope(name, default_name='graph pooling') as scope:
            self.current_V,self.current_A = make_graph_pooling_layer(self.current_V, self.current_A, self.current_Ps[Pindex], name)
        return self.current_V,self.current_A

    def make_graph_maxpooling_layer(self, Pindex, name=None):
        with tf.variable_scope(name, default_name='graph pooling') as scope:
            self.current_V,self.current_A = make_graph_maxpooling_layer(self.current_V, self.current_A, self.current_Ps[Pindex], name)
        return self.current_V,self.current_A

    def make_graph_unpooling_layer(self, Pindex, name=None):
        with tf.variable_scope(name, default_name='graph pooling') as scope:
            self.current_V,self.current_A = make_graph_unpooling_layer(self.current_V, self.current_A, self.current_Ps[Pindex], name)
        return self.current_V,self.current_A
