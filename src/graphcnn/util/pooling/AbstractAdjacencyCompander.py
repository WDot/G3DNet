from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractAdjacencyCompander(object):
    __metaclass__ = ABCMeta

    def __init__(self,V,A):
        self.V = V
        self.A = A
        self.N = V.shape[0]
        self.numDirs = 8
        self.flatA = 0

    @abstractmethod
    def contractA(self):
        pass
    @abstractmethod
    def expandA(self):
        pass

    def update(self,P):
        self.flatA = np.dot(np.dot(P.transpose(),self.flatA),P)
        self.V = np.dot(P.transpose(),self.V)
        self.N = self.V.shape[0]

