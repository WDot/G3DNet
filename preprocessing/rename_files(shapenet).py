import glob2
import numpy as np
import nltk
from nltk.corpus import wordnet
import os
import shutil
import pdb
import csv
import pcl

src_directory = '/shared/kgcoe-research/mil/modelnet/shapenet_500_pcd/'

synsetid = []
split=[]
ids=[]
modelid=[]
ids_dict = {}

with open('all.csv', 'rb') as f:
    reader = csv.reader(f) #csv read object file
    next(reader) # skip the headers
    for row in reader:
		ids.append(row[0])
		synsetid.append(row[1])
		modelid.append(row[3])
		split.append(row[4])
#pdb.set_trace()

syns = list(wordnet.all_synsets())
offsets_list = [(s.offset(), s) for s in syns]
offsets_dict = dict(offsets_list)
	
class_ids = set(synsetid)
class_ids = list(class_ids)
class_dict= {}
		
for id in class_ids:
	#pdb.set_trace()
	key=int(id)
	class_name = offsets_dict[key]
	class_name = str(class_name)
	value = class_name.split('.')[0][8:]
	class_dict[key] = value
		
		
path_all_files = '/shared/kgcoe-research/mil/modelnet/shapenet_500_pcd'
all_files = glob2.glob(path_all_files+'/**/*.pcd')
dest_directory = '/home/rnd7528/git/shapnet/shapenet_dataset'
i=0
j=0
#pdb.set_trace()

for file in all_files:
	#i=i+1
	print i
	pcd_file = pcl.load(file)
	name = file.split('/')
	try: 
		idx = modelid.index(name[-3])
		split_val = split[idx]
		name_id = ids[idx]
	except:
		#pdb.set_trace()
		#j=j+1
		continue
	class_name = class_dict[int(name[-4])]	
	dest_path = dest_directory + '/' + class_name + '/' + split_val
	if os.path.isdir(dest_path) == False:
		os.makedirs(dest_path)
	dest_file = dest_path + '/' + name_id + '.pcd'
	pcl.save(pcd_file,dest_file)
	#pdb.set_trace()

#pdb.set_trace()