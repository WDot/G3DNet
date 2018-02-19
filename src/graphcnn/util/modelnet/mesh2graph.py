import plyfile
import os
import os.path
import sys
import numpy
import itertools
import numpy.linalg
import scipy.io
import scipy.sparse
import GraphData as GD

NUM_DIR = 8

def ply2graph(plyPath):
    plydata = plyfile.PlyData.read(plyPath)
    graphData = GD.GraphData(NUM_DIR)
    for i in range(plydata['vertex'].count):
        graphData.addVertex(plydata['vertex'][i])
    graphData.lockVertices()
    for i in range(plydata['face'].count):
        # print(plydata['face'][i])
        graphData.addFace(plydata['face'][i])
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
