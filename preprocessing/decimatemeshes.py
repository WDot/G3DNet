
import os
import os.path
import sys
import subprocess
from subprocess import STDOUT, check_output
import ctypes
SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
CREATE_NO_WINDOW = 0x08000000    # From Windows API


#SCRIPT_PATH = 'decimatemeshes.mlx'
SCRIPT_PATH = './preprocessing/pointcloudify.mlx'



def Main():
    if len(sys.argv) > 2 and os.path.isdir(sys.argv[1]):
        counter = 0
        ignored = 0
        garbage = 0
        for root, subdirs, files in os.walk(sys.argv[1]):
            subDirectory = root[len(sys.argv[1]):]
            for file in files:
                fname, ext = os.path.splitext(file)
                inputPath = root + '/' + file
                outputPath = sys.argv[2] + '/' + subDirectory + '/' + fname + '.ply'
                if not os.path.isdir(sys.argv[2] + '/' + subDirectory):
                    os.makedirs(sys.argv[2] + '/' + subDirectory)
                if ext == '.off'\
                    and not os.path.isfile(outputPath)\
                    and os.stat(inputPath).st_size > 0:
                    scriptStr = 'C:\Program Files\VCG\MeshLab\meshlabserver.exe -i \"{0}\" -o \"{1}\" -s \"{2}\"'.format(\
                        inputPath, outputPath,SCRIPT_PATH)
                    print(scriptStr)
                    try:
                        process = subprocess.Popen(scriptStr,creationflags=CREATE_NO_WINDOW, )
                        output = check_output(scriptStr, stderr=STDOUT, timeout=120)
                        process.wait()
                        counter += 1
                        print('{0} files finished processing'.format(counter))
                    except:
                        print("File Ignored!!")
                        ignored += 1
                        continue
                elif os.stat(inputPath).st_size == 0:
                    print('Empty File, garbage!')
                    garbage += 1
    else:
        print('Could not find path!')
    print("Total Files converted = %d" %counter)
    print("Total Files ignored = %d" % ignored)
    print("Total Empty Files  = %d" % garbage)

Main()
