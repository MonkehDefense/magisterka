import math
import scipy.io
import pandas as pd
import csv
import os
from numpy.lib.arraysetops import unique
import numpy as np
import matplotlib
#import gc
#import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def main():
	TDL, VDL, mapa = loader()

	# print(len(TDL))
	# print(len(VDL))
	print(mapa)
	for rec, label in TDL:
		print(label[0].item())
		# print(type(label[0].item()))
		# print(type(int(1)))
		print(mapa[label[0].item()])
		
		# print(rec[0])
		break
	



def loader(train_dir = 'training2017', batch_size = 100, split_point = .2):
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
	inv_map = dict()
	# l = len(uniq)

	for i in range(len(uniq)):
		# t = [0]*l
		# t[i] = 1
		label_map[uniq[i]] = i
		inv_map[i] = uniq[i]

	
	
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
	# labs = torch.reshape(torch.as_tensor(train.label, dtype=torch.float32), (len(train.label),len(train.label[0])))
	labs = torch.reshape(torch.as_tensor(train.label, dtype=torch.int64), (len(train.label),))

	train_ds = TensorDataset(recs, labs)

	split_point = int(split_point * len(train_ds))
	train_ds, valid_ds = random_split(
		train_ds,
		[split_point, len(train_ds) - split_point],
		torch.Generator().manual_seed(42))


	train_dl = DataLoader(train_ds, batch_size, shuffle = True)
	valid_dl = DataLoader(valid_ds, batch_size, shuffle = True)
	
	return train_dl, valid_dl, inv_map


if __name__ == '__main__':
	main()
