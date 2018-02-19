import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from graphcnn.experiments.experiment_pcd import *
import pdb
import argparse

class Modelnet10Experiment():
    def __init__(self, architecture):
        self.arch = architecture.split(',')
        self.num_layers = len(self.arch)
    def create_network(self, net, input):
        net.create_network(input)
        conv_no = 1
        resnet_conv_no = 1
        pool_no = 1
        fc = 1
        gp = 1
        gup = 1
        gmp = 1
        convolution_type = self.arch[0]
        if convolution_type == 'OC':
            make_conv = net.make_graphcnn_layer
            make_conv_resnet_or_densenet = net.make_graphcnn_resnet_or_densenet_layer
        elif convolution_type =='UC':
            make_conv = net.make_graphcnn_unbiased_layer
            make_conv_resnet_or_densenet = net.make_graphcnn_unbiased_resnet_or_densenet_layer

        for i in xrange(1, self.num_layers):
            config = self.arch[i].split('_')
            if config[0]=='c':
                make_conv(int(config[1]),stride=int(config[2]), order=int(config[3]),name='conv'+str(conv_no), with_bn=True, with_act_func=True)
                conv_no+=1
            elif config[0]=='p':
                net.make_graph_embed_pooling(no_vertices=int(config[1]), name='pool1'+str(pool_no), with_bn=False, with_act_func=False)
                pool_no+=1
            elif config[0]=='fc':
                net.make_fc_layer(int(config[1]),name='fc'+str(fc), with_bn=int(config[2]), with_act_func=int(config[3]))
                fc+=1
            elif 'rc' in config[0]:
                if config[0][-1] == '0' :
                    name_layer = 'resnet_conv_block'
                elif config[0][-1] == '1' :
                    name_layer = 'densenet_conv_block'
                make_conv_resnet_or_densenet(no_filters=[int(j) for j in config[1].split('-')]  , stride=[int(j) for j in config[2].split('-')], order=[int(j) for j in config[3].split('-')],name=name_layer+str(resnet_conv_no), with_bn=[bool(j) for j in config[4].split('-')], with_act_func=[bool(j) for j in config[5].split('-')])
                resnet_conv_no+=1
            elif config[0]=='gp':
                net.make_graph_pooling_layer(int(config[1]), name='gp' + str(gp))
                gp += 1
            elif config[0]=='gmp':
                net.make_graph_maxpooling_layer(int(config[1]), name='gmp' + str(gp))
                gmp += 1
            elif config[0] == 'gup':
                net.make_graph_unpooling_layer(int(config[1]), name='gup' + str(gup))
                gup += 1

        #net.make_graph_embed_pooling(32, with_bn=False, with_act_func=False)
        # net.make_fc_layer(32,name='fc1', with_bn=True, with_act_func=True)
        # net.make_fc_layer(10,name='final', with_bn=False, with_act_func=False)

parser = argparse.ArgumentParser(description='Process input architecture')
parser.add_argument('--arch', default='OC,c_16_1_1,c_16_1_1,c_16_1_1,c_16_1_1,p_16,fc_10_0_0', help='Defines the model')  
parser.add_argument('--date', default='Sept02', help='Data run model')
parser.add_argument('--dataset_name', default='Modelnet10', help='Dataset name')
#Add loading pretrained weights option
parser.add_argument('--loading_weights_flag', default=0, type=int,help='loading weights flag')

parser.add_argument('--path_pretrained_weights', default='/home/thn2079/git/GraphCNN_WACV/GraphCNN/Graph-CNN/snapshots/Modelnet10-Oct6/Modelnet10-OC-c_16_1_1-c_16_1_1-c_16_1_1-c_16_1_1-p_16-fc_10_0_0-l2=0.0-l1=0.0/model-7', help='Path to pretrained weights')
parser.add_argument('--arch_loading', default='OC,c_16_1_1,c_16_1_1,c_16_1_1,c_16_1_1', help='Specific architecture to load weights from')
# parser.add_argument('--test_file', default=TEST_FILE, help='test file path')
parser.add_argument('--train_flag', default=1, type=int,help='training flag')
parser.add_argument('--debug_flag', default=0, type=int,help='debugging flag, if set as true will not save anything to summary writer')
parser.add_argument('--num_iter', default=4000, type=int,help='Number of iterations')
##feature_type==1: binary feature
##feature_type==0: 6 features
parser.add_argument('--feature_type', default=1, type=int,help='Feature type flag for modelnet')
###If feature_type==0,we need k nearest neighbor
parser.add_argument('--K', default=6, type=int,help='K nearest neighbors')
parser.add_argument('--num_vertices', default=516, type=int,help='Number of vertices in the graph')
parser.add_argument('--num_channels', default=3, type=int,help='Number of channels')
parser.add_argument('--num_classes', default=10, type=int,help='Number of classes')

parser.add_argument('--train_batch_size', default=60, type=int,help='Batch size for training')
parser.add_argument('--test_batch_size', default=50, type=int,help='Batch size for testing')
parser.add_argument('--snapshot_iter', default=200, type=int,help='Take snapshot each number of iterations')
parser.add_argument('--starter_learning_rate', default=0.01, type=float,help='Started learning rate')
parser.add_argument('--learning_rate_step', default=1000, type=int,help='Learning rate step decay')
parser.add_argument('--learning_rate_exp', default=0.1, type=float,help='Learning rate exponential')
parser.add_argument('--optimizer', default='adam', help='Choose optimizer type')
parser.add_argument('--iterations_per_test', default=4000, type=int,help='Test model by validation set each number of iterations')
parser.add_argument('--display_iter', default=5, type=int,help='Display training info each number of iterations')
parser.add_argument('--l2',default=0.0,type=float,help="L2 Regularization parameter")
parser.add_argument('--l1',default=0.0,type=float,help="L1 Regularization parameter")
parser.add_argument('--pool_ratios',default='0.5_0.5_0.5',help="Ratio of vertex reductions for each pooling")
parser.add_argument('--cluster_alg',default='Lloyd',help='How should pooling cluster vertices?')
parser.add_argument('--prefix',default='/home/data/',help='Prefix of Data Location')
parser.add_argument('--dataset',default='modelnet10',help='Dataset to train')
parser.add_argument('--group_name',default='WACV2018',help='Experiment Directory Name')
parser.add_argument('--trial_name',default='G3DNet18',help='Experiment Directory Name')

args = parser.parse_args()

if args.dataset == 'modelnet10':
    TRAIN_FILE = './preprocessing/modelnet10_trainval.csv'
    TEST_FILE = './preprocessing/modelnet10_test.csv'
    # TEST_FILE = './preprocessing/modelnet10_test.csv'
    LABEL_FILE = './preprocessing/modelnet10_labels.csv'
    #MODELNET10_NUM_CLASSES = 10
elif args.dataset == 'modelnet40':
    TRAIN_FILE = './preprocessing/modelnet40_auto_aligned_trainval.csv'
    TEST_FILE = './preprocessing/modelnet40_auto_aligned_test.csv'
    # TEST_FILE = './preprocessing/modelnet40_test.csv'
    LABEL_FILE = './preprocessing/modelnet40_labels.csv'
    #MODELNET10_NUM_CLASSES = 40
elif args.dataset == 'modelnetfull':
    #There is no such thing as separate train/val sets for modelnetFull, this is just
    #pure extra samples to learn from
    TRAIN_FILE = './preprocessing/modelnetFull_500_train.csv'
    TEST_FILE = './preprocessing/modelnetFull_500_train.csv'
    LABEL_FILE = './preprocessing/modelnetFull_labels.csv'
    #MODELNET10_NUM_CLASSES = 421
elif args.dataset == 'shapenetcore':
    #We don't predict on Shapenet, so we use both train and val for more samples
    TRAIN_FILE = './preprocessing/shapenet_500_trainval.csv'
    TEST_FILE = './preprocessing/shapenet_500_test.csv'
    LABEL_FILE = './preprocessing/shapenet_labels.csv'
    #MODELNET10_NUM_CLASSES = 55
elif args.dataset == 'sydney0':
    #There is no such thing as separate train/val sets for modelnetFull, this is just
    #pure extra samples to learn from
    TRAIN_FILE = './preprocessing/sydneyfoldn0.csv'
    TEST_FILE = './preprocessing/sydneyfold0.csv'
    LABEL_FILE = './preprocessing/sydney_labels.csv'
    #MODELNET10_NUM_CLASSES = 14
elif args.dataset == 'sydney1':
    #There is no such thing as separate train/val sets for modelnetFull, this is just
    #pure extra samples to learn from
    TRAIN_FILE = './preprocessing/sydneyfoldn1.csv'
    TEST_FILE = './preprocessing/sydneyfold1.csv'
    LABEL_FILE = './preprocessing/sydney_labels.csv'
    #MODELNET10_NUM_CLASSES = 14
elif args.dataset == 'sydney2':
    #There is no such thing as separate train/val sets for modelnetFull, this is just
    #pure extra samples to learn from
    TRAIN_FILE = './preprocessing/sydneyfoldn2.csv'
    TEST_FILE = './preprocessing/sydneyfold2.csv'
    LABEL_FILE = './preprocessing/sydney_labels.csv'
    #MODELNET10_NUM_CLASSES = 14
elif args.dataset == 'sydney3':
    #There is no such thing as separate train/val sets for modelnetFull, this is just
    #pure extra samples to learn from
    TRAIN_FILE = './preprocessing/sydneyfoldn3.csv'
    TEST_FILE = './preprocessing/sydneyfold3.csv'
    LABEL_FILE = './preprocessing/sydney_labels.csv'
    #MODELNET10_NUM_CLASSES = 14


if args.feature_type==1:
    MODELNET10_L = 8
elif args.feature_type==0:
    MODELNET10_L = args.K * 6
MODELNET10_N = args.num_vertices
MODELNET10_C = args.num_channels
MODELNET10_NUM_CLASSES = args.num_classes

poolRatiosList = [float(x) for x in args.pool_ratios.split('_')]

exp = GraphCNNPCDExperiment(args.group_name, args.trial_name, Modelnet10Experiment(args.arch),args.prefix,MODELNET10_N,MODELNET10_C, \
                               MODELNET10_L, args.K, MODELNET10_NUM_CLASSES, TRAIN_FILE, TEST_FILE, LABEL_FILE,\
                            args.train_flag, args.l2, args.l1, args.feature_type,poolRatiosList,\
                               args.path_pretrained_weights,args.cluster_alg)



exp.num_iterations = args.num_iter
exp.optimizer = args.optimizer
exp.debug = bool(args.debug_flag)
exp.train_batch_size = args.train_batch_size
exp.test_batch_size = args.test_batch_size
exp.loading_weights_flag = bool(args.loading_weights_flag)
exp.arch_loading=args.arch_loading
exp.crop_if_possible = True
exp.snapshot_iter = args.snapshot_iter
exp.learning_rate_step = args.learning_rate_step
exp.starter_learning_rate = args.starter_learning_rate
exp.learning_rate_exp = args.learning_rate_exp
exp.iterations_per_test = args.iterations_per_test
exp.display_iter = args.display_iter
        
#exp.preprocess_data(dataset)

acc, std = exp.run()
print_ext('Result: %.4f (+- %.4f)' % (acc, std))
