import sys
import os
import os.path
import plyfile
import numpy
import pcl

NUM_DIR = 6

def ply2array(plyPath):
    plydata = plyfile.PlyData.read(plyPath)
    vertices = []
    #vertices = numpy.array(plydata['vertex'][0])
    for i in range(plydata['vertex'].count):
        vertices.append(list(plydata['vertex'][i]))
    vertices = numpy.array(vertices)
    return vertices

def ply2structarray(plyPath):
    plydata = plyfile.PlyData.read(plyPath)
    vertices = numpy.array(plydata['vertex'][0],dtype=[('x','f4'),('y','f4'),('z','f4')])
    for i in range(1,plydata['vertex'].count):
        vertices = numpy.append(vertices,numpy.array([plydata['vertex'][i]],dtype=vertices.dtype))
    return vertices


def Main():
    if len(sys.argv) > 3 and os.path.isdir(sys.argv[1]):
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
                outputPath = sys.argv[2] + '/' + subDirectory + '/' + fname + '.' + sys.argv[3]
                if not os.path.isdir(sys.argv[2] + '/' + subDirectory):
                    os.makedirs(sys.argv[2] + '/' + subDirectory)
                if ext == '.ply' and not os.path.isfile(outputPath):
                    if sys.argv[3] == 'npy':
                        vertices = ply2structarray(inputPath)
                        numpy.save(outputPath,vertices)
                    elif sys.argv[3] == 'pcd':
                        vertices = ply2array(inputPath)
                        print(vertices.shape)
                        pc = pcl.PointCloud(vertices)
                        pcl.save(pc,outputPath)
                    print('Saved {0}'.format(outputPath))

    else:
        print('Could not find path!')

Main()
