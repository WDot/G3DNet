from .AbstractAdjacencyCompander import AbstractAdjacencyCompander
import numpy as np
import sys
import os
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../../preprocessing')))
from mnist_to_graph_tensor import mnist_adj_mat

class ImageAdjacencyCompander(AbstractAdjacencyCompander):
    def __init__(self,V,A):
        super(ImageAdjacencyCompander, self).__init__(V, A)
        self.NUM_DIRS = 8

    def contractA(self):
        self.flatA = self.A.sum(axis=1)
        return self.flatA

    def expandA(self):
        return self.A

    def update(self, P):
        Ptiled = np.tile(np.expand_dims(P,axis=1),(1,self.NUM_DIRS,1))
        Ptranspose = np.transpose(Ptiled, axes=[1, 2, 0])
        Pnottranspose = np.transpose(Ptiled, axes=[1, 0, 2])
        Abatched = np.transpose(self.A, axes=[1, 0, 2])
        leftMultiply = np.matmul(Ptranspose, Abatched)
        rightMultiply = np.matmul(leftMultiply, Pnottranspose)
        self.A = np.transpose(rightMultiply, axes=[1, 0, 2])
        self.V = np.dot(P.transpose(), self.V)
        self.N = self.V.shape[0]