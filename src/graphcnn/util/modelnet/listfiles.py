import os
import os.path
import sys

def Main():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        with open('testData516.txt','w') as testFile, open('trainData516.txt','w') as trainFile:
            for root, subdirs, files in os.walk(sys.argv[1]):
                subDirectory = root[len(sys.argv[1]):]
                for file in files:
                    fname, ext = os.path.splitext(file)
                    inputPath = root + '/' + file
                    if 'test' in inputPath:
                        testFile.write(inputPath + '\n')
                    elif 'train' in inputPath:
                        trainFile.write(inputPath + '\n')

    else:
        print('Could not find path!')


Main()
