import numpy
import itertools
import scipy
import numpy.random
import math
import transforms3d

class GraphData(object):
    def __init__(self,numDir):
        self.vertexCount = 0
        self.V = []
        self.vertexLock = False
        self.numDir = numDir

    def addVertex(self, vertex):
        if not self.vertexLock:
            # print(numpy.array((vertex['x'],vertex['y'],vertex['z'])))
            self.vertexCount += 1
            self.V.append(numpy.array((vertex['x'], vertex['y'], vertex['z'])))

    def dropVertices(self,p):
        removedIndices = numpy.random.choice(range(self.vertexCount), size=int(math.ceil(p * self.vertexCount)), replace=False)
        for index in sorted(removedIndices,reverse=True):
            self.V.pop(index)
        self.vertexCount -= len(removedIndices)

    def lockVertices(self):
        self.vertexLock = True
        self.A = []
        for i in range(self.numDir):
            self.A.append(numpy.zeros((self.vertexCount, self.vertexCount)))

    def normalizeVertices(self):
        mean = numpy.mean(self.V,axis=0)
        std = numpy.std(self.V,axis=0)
        if not std.all():
            M = numpy.eye(3)
            angle = numpy.random.uniform(0.001, 0.1, size=3)
            sign = -1 * numpy.random.choice([0, 1], size=3, replace=True)
            M = numpy.dot(transforms3d.axangles.axangle2mat([0, 0, 1], sign[0] * angle[0]), M)
            M = numpy.dot(transforms3d.axangles.axangle2mat([0, 1, 0], sign[1] * angle[1]), M)
            M = numpy.dot(transforms3d.axangles.axangle2mat([1, 0, 0], sign[2] * angle[2]), M)
            for i in range(len(self.V)):
                self.V[i] = numpy.dot(self.V[i], M.T)

            mean = numpy.mean(self.V, axis=0)
            std = numpy.std(self.V, axis=0)
        for i in range(len(self.V)):
            self.V[i] -= mean
            self.V[i] /= std

            # Inspired by Python's "pairwise" recipe, but creates a cycle

    def __edgeIter(self, face):
        faceCycle = numpy.append(face[0], face[0][0])
        a, b = itertools.tee(faceCycle)
        next(b, None)
        return zip(a, b)

    def addEdge(self,vertex1,vertex2,edgeFeature):
        zindex = numpy.dot([4, 2, 1], numpy.greater((self.V[vertex1] - self.V[vertex2]), numpy.zeros(3)));
        edgeLen = 1
        # print('From {0} to {1}: Len {2}',i,j,edgeLen)
        self.A[zindex][vertex1, vertex2] = edgeFeature
        self.A[zindex][vertex2, vertex1] = edgeFeature
        #print('Edge Feature: {0}'.format(edgeFeature))

    def addFace(self, face):
        for i, j in self.__edgeIter(face):
            # edgeLen = 1./numpy.linalg.norm(self.V[i] - self.V[j],2).astype(numpy.float32)
            # Let's try this for simplicity, the information should theoretically
            # be in the nodes
            # Ignore edges on the -z axis
            #if self.V[i][2] >= 0 and self.V[j][2] >= 0:
            zindex = numpy.dot([4, 2, 1], numpy.greater((self.V[i] - self.V[j]), numpy.zeros(3)));
            edgeLen = 1
            # print('From {0} to {1}: Len {2}',i,j,edgeLen)
            self.A[zindex][i, j] = edgeLen
            self.A[zindex][j, i] = edgeLen

    def flattenA(self):
        Aout = numpy.zeros((self.vertexCount,self.vertexCount))
        for i in range(self.numDir):
            Aout += self.A[i]
        return Aout




    def saveAsMat(self, fileOut):
        # Need to compress otherwise each sample is way too big!
        #for i in range(self.numDir):
        #    self.A[i] = scipy.sparse.coo_matrix(self.A[i])
        scipy.io.savemat(fileOut, mdict={'vCount': self.vertexCount, 'V': numpy.asarray(self.V, dtype=numpy.float32),
                                         'A': self.A}, do_compression=True)

    def loadFromMat(self,fileIn):
        inDict = scipy.io.loadmat(fileIn)
        self.V = []
        for v in range(inDict['vCount']):
            self.V.append(inDict['V'][v][:])

        [_,L] = inDict['A'].shape

        self.A = []
        for l in range(L):
            print(inDict['A'][l].shape)
            self.A.append(inDict['A'][l][:][:])

    def plotGraph(self):
        pass
