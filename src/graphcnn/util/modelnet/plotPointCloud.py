import sys
import os
import os.path
import plyfile
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def ply2structarray(plyPath):
    plydata = plyfile.PlyData.read(plyPath)
    vertices = numpy.array(plydata['vertex'][0],dtype=[('x','f4'),('y','f4'),('z','f4')])
    for i in range(1,plydata['vertex'].count):
        vertices = numpy.append(vertices,numpy.array([plydata['vertex'][i]],dtype=vertices.dtype))
    return vertices

def Main():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        points = ply2structarray(sys.argv[1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points['x'], points['y'], points['z'], c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    else:
        print('Could not find path!')

Main()