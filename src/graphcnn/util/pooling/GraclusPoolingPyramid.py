from .AbstractPoolingPyramid import AbstractPoolingPyramid
import scipy.sparse
import pyamg
import numpy as np
#from graphcnn.util.modelnet.pointCloud2Graph import ply2graph
import tensorflow as tf
#import matlab.engine
import sys
import os
import os.path
#import matlab
import scipy.sparse
import time
import datetime
import subprocess
from subprocess import STDOUT, check_output

class GraclusPoolingPyramid(AbstractPoolingPyramid):

    def __init__(self,numRepresentations,companderConstructor, ratios):#, matlabEngine):
        super(GraclusPoolingPyramid, self).__init__(numRepresentations,companderConstructor)
        self.ratios = ratios
        #self.eng = matlabEngine

    #Assumes normalized cut
    #k is number of clusters
    def GraclusByHand(self,A,k):
        numVertices = A.shape[0]
        #Initial labels
        pi = np.random.randint(0,k,size=numVertices)
        #Weights of each vertex
        w = np.sum(A,axis=1)
        #Diagonal degree matrix
        D = np.diag(w)
        #print(D)
        #Should be high enough so that K is positive definite?
        #This essentially says any weighted sum z^T*A*z will always be positive for any z
        #Heck if I know how to do that
        sigma = 1
        wInv = w
        wInv[wInv == 0] = 1
        Dinv = np.diag(1/wInv)
        #Dinv = np.linalg.pinv(D)
        #print(Dinv)
        #Kernel matrix
        #The kernel matrix entries Kij are basically the kernel function phi applied to phi(a_i)phi(a_j)
        K = sigma * Dinv + np.dot(np.dot(Dinv,A),Dinv)
        ignoreDiagsK = np.invert(np.eye(K.shape[0]).astype(np.bool)).astype(np.float32)
        KignoreDiags = K * ignoreDiagsK
        #print(K)
        #Should technically check for convergence, but since I'm hacking I'll just set it to 10 every time
        tmax = 10
        for t in range(tmax):
            piOld = pi
            d = 1000000000*np.ones((A.shape[0],k))
            #Calculate distortion(cost)
            for c in range(k):
                i = np.arange(A.shape[0])
                j = np.where(pi == c)[0]
                l = np.where(pi == c)[0]
                if j.size > 0:
                    wjsum = np.sum(w[j])
                    if wjsum > 0:
                        jv, lv = np.meshgrid(j,l)
                        term1 = K[i,i]
                        term2 = 2 * np.dot(w[j],KignoreDiags[np.ix_(i,j)].transpose()) / wjsum
                        #print(w[jv])
                        #print(w[lv])
                        #print(K[jv,lv])
                        #print(w[jv]*w[lv]*K[jv,lv])
                        ignoreDiags = np.invert(np.eye(len(j)).astype(np.bool)).astype(np.float32)

                        term3 = np.sum(w[jv]*w[lv]*K[jv,lv]*ignoreDiags) / (wjsum*wjsum)
                        #Calculate mc for reals
                        #NOT d(i,c), d(i,mc)!
                        d[i,c] = term1 - term2 + term3
                        #if np.isnan(d).any():
                        #    print('WHAAAAT')
            #Find minimum cost for each vertex i
            #print(pi)
            #print(d)
            pi = np.argmin(d,axis=1)
            if (pi == piOld).all():
                print('Number of Iterations: {0}'.format(t))
                break
        return pi

    def makeP(self,A,V=None):
        Plist = []
        companderInstance = self.companderConstructor(V,A)

        for pIndex in range(self.numRepresentations):
            outSize = int(np.floor(self.ratios[pIndex]*A.shape[0]))
            flatA = companderInstance.contractA()
            t = time.time()
            labels = self.GraclusByHand(flatA,outSize)
            print(labels)
            elapsed = time.time() - t
            print('Time Elapsed: {0}'.format(elapsed))
            #filename = datetime.datetime.now().strftime('adjacency-%Y%m%d-%H%M%S')
            #self.writeGraclusFile(flatA, filename)
            #scriptStr = '../util/pooling/graclus1.2/graclus.exe ' + filename + ' ' + str(outSize)
            #process = subprocess.Popen(scriptStr,stdout=STDOUT)
            #output = check_output(scriptStr, stderr=STDOUT, timeout=120)
            #process.wait()
            #sys.exit()

            #labels = self.eng.graclus(matlab.double(companderInstance.contractA().tolist()),outSize)
            #P = pyamg.aggregation.aggregate.lloyd_aggregation(\
            #scipy.sparse.csr_matrix(companderInstance.contractA()),ratio=self.ratios[pIndex],distance='same',maxiter=10)[0]
            labels = np.squeeze(np.array(labels).astype(np.int32) - 1)
            #print(labels)
            P = np.zeros((A.shape[0],outSize))
            P[np.arange(A.shape[0]),labels] = 1
            #print('Nonzero P: {0}'.format(np.count_nonzero(P)))
            Pcolsum = np.tile(np.count_nonzero(P,axis=0),(P.shape[0],1))
            Pcolsum[Pcolsum == 0] = 1
            P = np.divide(P,Pcolsum.astype(np.float64))
            Plist.append(P.astype(np.float32))
            #print(P.shape)
            companderInstance.update(P)
            A = companderInstance.expandA()
            V = companderInstance.V
        return Plist

    def write(self,Ps,As):
        AsparseList = []
        for A in As:
            currentA = A.tolist()
            pass
