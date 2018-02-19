import sys
import os
import os.path
import plyfile
import numpy
import scipy
import scipy.spatial
import scipy.io
import GraphData as GD

def Main():
    if len(sys.argv) > 2 and os.path.isfile(sys.argv[1]):
        graphData = GD.GraphData(int(sys.argv[2]))
        graphData.loadFromMat(sys.argv[1])


    else:
        print('Could not find path!')

Main()

