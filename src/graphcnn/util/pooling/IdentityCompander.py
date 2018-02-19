from .AbstractAdjacencyCompander import AbstractAdjacencyCompander
import numpy as np

class IdentityCompander(AbstractAdjacencyCompander):
    def __init__(self,V,A):
        super(IdentityCompander, self).__init__(V, A)

    def contractA(self):
        self.flatA = self.A
        return self.flatA

    def expandA(self):
        self.A = self.flatA
        return self.A
