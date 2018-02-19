import sys
import os
import os.path
from .AbstractPoolingPyramid import AbstractPoolingPyramid
from .GraclusPoolingPyramid import GraclusPoolingPyramid
from .LloydPoolingPyramid import LloydPoolingPyramid
from .SpectralClusteringPoolingPyramid import SpectralClusteringPoolingPyramid
#import matlab.engine

class PoolingFactory():

    #def __init__(self):
     #   pass
        #self.eng = matlab.engine.start_matlab()
        #self.eng.addpath(os.path.abspath(os.path.join(os.path.dirname(__file__),'./graclus1.2/matlab/')))

    def CreatePoolingPyramid(self,numRepresentations, companderConstructor, ratios, id='Lloyd'):
        if id == 'Lloyd':
            return LloydPoolingPyramid(numRepresentations,companderConstructor,ratios)
        elif id == 'Spectral':
            return SpectralClusteringPoolingPyramid(numRepresentations,companderConstructor,ratios)
        elif id == 'Graclus':
            return GraclusPoolingPyramid(numRepresentations,companderConstructor,ratios)#,self.eng)
