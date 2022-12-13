import math
import scipy.io
import pandas as pd
import csv
import os
from numpy.lib.arraysetops import unique
import numpy as np
import matplotlib
import gc
#import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

def main():
	DL, map = loader()
	print(map)
	for rec, label in DL:
		print(label[0])
		print(rec[0])
		break
	



def loader(train_dir = 'training2017', batch_size = 100):
	records = []
	rec_id = []
	labels_set = []
	lab_id = []

	for filename in os.listdir(train_dir):
		if not filename.endswith('.mat'):
			continue

		f_path = os.path.join(train_dir, filename)
		record = scipy.io.loadmat(f_path)['val'][0]
	
		if len(record) >= 8192:
			records.append(record[:8192])
			rec_id.append(filename[:-4])
			
			

	labels_csv = os.path.join(train_dir, 'REFERENCE-v3.csv')
	
	with open(labels_csv, newline='') as file:
		for row in csv.reader(file, delimiter=' ', quotechar='|'):
			labels_set.append(row[0][-1])
			lab_id.append(row[0][:-2])


	label_map = dict()
	uniq = unique(labels_set)
	l = len(uniq)

	for i in range(l):
		t = [0]*l
		t[i] = 1
		label_map[uniq[i]] = t
	
	labels_set = [label_map[i] for i in labels_set]

	train = pd.DataFrame({'key': rec_id, 'record': records})
	lab = pd.DataFrame({'key': lab_id, 'label': labels_set})

	train = train.merge(lab, how = 'left', left_on = 'key', right_on = 'key')
	train.set_index('key').sort_index()
	

	# konwersja danych na tensory i stworzenie dataloadera

	
	recs = torch.reshape(torch.as_tensor(train.record, dtype=torch.float32),
					(len(train.record),1,len(train.record[0])))
	#recs = torch.zeros(len(train.record),1,len(train.record[0]))
	#recs[:,0] = torch.as_tensor(train.record)
	labs = torch.reshape(torch.as_tensor(train.label, dtype=torch.float32),
					 (len(train.label),1,len(train.label[0])))

	train_ds = TensorDataset(recs, labs)
	train_dl = DataLoader(train_ds, batch_size, shuffle = True)
	
	return train_dl, label_map


if __name__ == '__main__':
	main()
