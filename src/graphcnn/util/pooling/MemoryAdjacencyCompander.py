from .AbstractAdjacencyCompander import AbstractAdjacencyCompander
import numpy as np

class MemoryAdjacencyCompander(AbstractAdjacencyCompander):

    def __init__(self,V,A):
        super(MemoryAdjacencyCompander, self).__init__(V, A)

    def contractA(self):
        self.flatA = self.A.sum(axis=1)
        return self.flatA

    def expandA(self):
        return self.A
    def update(self,P):
        #  print(P.shape)
        Aposcorrelation = np.asarray(np.dot(np.dot(P.transpose(),self.A[:,0,:].squeeze()),P))
        Anegcorrelation = np.asarray(np.dot(np.dot(P.transpose(),self.A[:,1,:].squeeze()),P))
        Azerocorrelation = np.asarray(np.dot(np.dot(P.transpose(),self.A[:,2,:].squeeze()),P))
        #print(Aposcorrelation.shape)
        #print(Anegcorrelation.shape)
        #print(Azerocorrelation.shape)
        #print(type(Aposcorrelation))
        self.A = np.stack((Aposcorrelation,Anegcorrelation,Azerocorrelation),axis=1).astype(np.float32)
        self.V = np.dot(P.transpose(),self.V).astype(np.float32)
        self.N = self.V.shape[0]
