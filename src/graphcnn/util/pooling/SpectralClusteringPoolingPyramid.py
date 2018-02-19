from .AbstractPoolingPyramid import AbstractPoolingPyramid
import scipy.sparse
import pyamg
import numpy as np
#from graphcnn.util.modelnet.pointCloud2Graph import ply2graph
import tensorflow as tf
import sklearn.cluster
import scipy.sparse
import time

class SpectralClusteringPoolingPyramid(AbstractPoolingPyramid):

    def __init__(self,numRepresentations,companderConstructor, ratios):
        super(SpectralClusteringPoolingPyramid, self).__init__(numRepresentations,companderConstructor)
        self.ratios = ratios

    def makeP(self,A,V=None):
        Plist = []
        companderInstance = self.companderConstructor(V,A)

        for pIndex in range(self.numRepresentations):
            outSize = int(np.floor(self.ratios[pIndex]*A.shape[0]))
            #t = time.time()
            numComponents = int(np.maximum(np.floor(outSize/4),1))
            labels = sklearn.cluster.spectral_clustering(scipy.sparse.csr_matrix(companderInstance.contractA()),\
                                            n_clusters=outSize,eigen_solver='arpack',n_init=1,n_components=numComponents)
            #elapsed = time.time() - t
            #print('Elapsed: {0}'.format(elapsed))
            #P = pyamg.aggregation.aggregate.lloyd_aggregation(\
            #scipy.sparse.csr_matrix(companderInstance.contractA()),ratio=self.ratios[pIndex],distance='same',maxiter=10)[0]

            P = np.zeros((A.shape[0],outSize))
            P[np.arange(A.shape[0]),labels] = 1
            #print('Nonzero P: {0}'.format(np.count_nonzero(P)))
            Pcolsum = np.tile(np.count_nonzero(P,axis=0),(P.shape[0],1))
            Pcolsum[Pcolsum == 0] = 1
            P = np.divide(P,Pcolsum.astype(np.float64))
            Plist.append(P.astype(np.float32))
            #print(P.shape)
            companderInstance.update(P)
            A = companderInstance.expandA()
            V = companderInstance.V
        return Plist

    def write(self,Ps,As):
        AsparseList = []
        for A in As:
            currentA = A.tolist()
            pass
