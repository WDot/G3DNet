#The usual Modelnet model header has a structure like this:
#OFF
#X Y Z
#Some of the Modelnet40 model headers have a structure like this:
#OFFX Y Z
#This makes MeshlabServer unhappy. This script basically adds a newline at the appropriate place.

import sys
import os
import os.path
import re

MATCH_HEADER = re.compile('(OFF)(([\d]+[\s]*)+)')

def FixHeader(inPath,outPath):
    with open(inPath) as f:
        lines = f.read().splitlines()
        matches = MATCH_HEADER.findall(lines[0])
        if len(matches) > 0:
            newLine1 = matches[0][0]
            newLine2 = matches[0][1]
            print(newLine1)
            print(newLine2)
            lines[0] = newLine1
            lines.insert(1,newLine2)
    with open(outPath,'w+') as f:
        for line in lines:
            f.write(line + '\n')




def Main():
    if len(sys.argv) > 2 and os.path.isdir(sys.argv[1]):
        counter = 0
        for root, subdirs, files in os.walk(sys.argv[1]):
            subDirectory = root[len(sys.argv[1]):]
            for file in files:
                fname, ext = os.path.splitext(file)
                inputPath = root + '/' + file
                outputPath = sys.argv[2] + '/' + subDirectory + '/' + fname + '.off'
                if not os.path.isdir(sys.argv[2] + '/' + subDirectory):
                    os.makedirs(sys.argv[2] + '/' + subDirectory)
                if ext == '.off' and not os.path.isfile(outputPath):
                    FixHeader(inputPath,outputPath)
                    counter += 1
                    #print('{0} files finished processing'.format(counter))
    else:
        print('Could not find path!')


Main()