import sys
import os
import os.path
import plyfile
import numpy as np
import csv

def sydneyToPly(inPath,outPath):
    vertices = []
    filename = os.path.basename(inPath)
    filename, ext = os.path.splitext(filename)
    with open(inPath,'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for row in csvReader:
            vertices.append((float(row[3]),float(row[4]),float(row[5])))
            #print(str(row[3]) + ' ' + str(row[4]) + ' ' + str(row[5]))
        vertices = np.array(vertices,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = plyfile.PlyElement.describe(vertices,'vertex',val_types={'vertex': 'f4'})
        plyfile.PlyData([el],text=True).write(outPath + '/' + filename + '.ply')

        #print(vertices.shape)

INPATH = 'C:/data/sydney-urban-objects-dataset/objects'
OUTPATH = 'C:/data/sydney_ply'
for item in os.listdir(INPATH):
    filename, ext = os.path.splitext(item)
    if ext == '.csv':
        #print(item)
        sydneyToPly(INPATH + '/' + item,OUTPATH)

