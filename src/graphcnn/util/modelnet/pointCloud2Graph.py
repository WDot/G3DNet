import sys
import os
import os.path
import plyfile
import numpy
import scipy
import scipy.spatial
import scipy.io
from .GraphData import GraphData as GD

NUM_DIR = 6

def ply2graph(plyPath):
    plydata = plyfile.PlyData.read(plyPath)
    graphData = GD.GraphData(NUM_DIR)
    for i in range(plydata['vertex'].count):
        graphData.addVertex(plydata['vertex'][i])
    graphData.lockVertices()
    graphData.normalizeVertices()
    kdtree = scipy.spatial.KDTree(graphData.V)
    #First nearest neighbor is always the point itself!
    knns = kdtree.query(graphData.V,k=(NUM_DIR+1))
    [vertexCount,edgeCount] = knns[0].shape
    for v1 in range(vertexCount):
        for v2 in range(1,edgeCount):
            graphData.addEdge(v1,knns[1][v1][v2],knns[0][v1][v2])
    return graphData

def Main():
    if len(sys.argv) > 2 and os.path.isdir(sys.argv[1]):
        # dirFiles = [f for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]
        if not os.path.isdir(sys.argv[2]):
            os.mkdir(sys.argv[2])
            # for dirFile in dirFiles:
        #	fname,ext = os.path.splitext(dirFile)
        for root, subdirs, files in os.walk(sys.argv[1]):
            subDirectory = root[len(sys.argv[1]):]
            for file in files:
                fname, ext = os.path.splitext(file)
                inputPath = root + '/' + file
                outputPath = sys.argv[2] + '/' + subDirectory + '/' + fname + '.mat'
                if not os.path.isdir(sys.argv[2] + '/' + subDirectory):
                    os.makedirs(sys.argv[2] + '/' + subDirectory)
                if ext == '.ply' and not os.path.isfile(outputPath):
                    graphData = ply2graph(inputPath)
                    graphData.saveAsMat(outputPath)
                    print('Saved {0}'.format(outputPath))

    else:
        print('Could not find path!')

Main()
