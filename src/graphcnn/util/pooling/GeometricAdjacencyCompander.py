from .AbstractAdjacencyCompander import AbstractAdjacencyCompander
import numpy as np

class GeometricAdjacencyCompander(AbstractAdjacencyCompander):

    def __init__(self,V,A):
        super(GeometricAdjacencyCompander, self).__init__(V, A)

    def contractA(self):
        self.flatA = self.A.sum(axis=1)
        return self.flatA

    def expandA(self):
        expandedA = np.zeros((self.N,self.numDirs,self.N))
        #print(self.N)
        #print(self.flatA.shape)
        (iVals,jVals) = np.nonzero(self.flatA)
        zindex = np.dot([4, 2, 1], np.greater((self.V[iVals,:] - self.V[jVals,:]).transpose(), np.zeros((3,iVals.shape[0]))));
        edgeLen = np.linalg.norm(self.V[iVals,:] - self.V[jVals,:],axis=1)
            # print('From {0} to {1}: Len {2}',i,j,edgeLen)
        expandedA[iVals, zindex, jVals] = edgeLen
        expandedA[jVals, zindex, iVals] = edgeLen
        self.A = expandedA


        return expandedA
