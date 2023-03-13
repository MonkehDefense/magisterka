# import math
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
	# loader()

	print(len(TDL))
	print(len(VDL))
	print(mapa)
	for rec, label in TDL:
		print(label[0].item())
		print(mapa[label[0].item()])
		print('batch:', rec.shape)
		print(rec[0])
		
		break
	














def loader(train_dir = 'training2017', batch_size = 100, split_point = .2, seed = None):
	records = []
	labels = []
	recId2LabId = dict()





	# mapa etykiet
	label_map = {'N':0,'A':1,'O':2,'~':3}
	inv_map = {0:'N',1:'A',2:'O',3:'~'}

	# załadowanie etykiet	
	labels_csv = os.path.join(train_dir, 'REFERENCE-v3.csv')
	with open(labels_csv, newline='') as file:
		for row in csv.reader(file, delimiter=',', quotechar='|'):
			recId2LabId[row[0]] = label_map[row[1]]

	# print(recId2LabId)



	# załadowanie danych
	for filename in os.listdir(train_dir):
		if not filename.endswith('.mat'):
			continue

		f_path = os.path.join(train_dir, filename)
		record = scipy.io.loadmat(f_path)['val'][0]
		
	
		if len(record) >= 4096:
			# zapisy o długości minimum 4096, skracane do 2048
			records.append(record[:4096:2])
			labels.append(recId2LabId[filename[:-4]])
	
	# print(records[0])

	# konwersja na tensory, dodanie kanału
	# konwersja na float, do obliczeń
	records = torch.from_numpy(np.stack(records)).type(torch.float)
	records = torch.unsqueeze(records,-2)
	labels = torch.tensor(labels)

	# print(records.shape)
	# print(labels.shape)


	# normalizacja
	# records = (records - mean)/sd
	# print(records.dtype)
	rec_mean = torch.mean(records, dim = 2, keepdim = True)
	rec_sd = torch.std(records, dim = 2, keepdim = True)

	records = (records - rec_mean)/rec_sd
	# print(records[0])


	# print(rec_mean.shape)
	# print(rec_sd.shape)
	# print(records[0])
	# print(rec_mean[0])
	# print((records - rec_mean.unsqueeze(1))[0])


	# Datasety
	train_ds = TensorDataset(records, labels)

	split_point = int(split_point*len(train_ds))
	if seed is not None:
		train_ds, valid_ds = random_split(
			train_ds,
			[split_point, len(train_ds) - split_point],
			torch.Generator().manual_seed(seed))
	else:
		train_ds, valid_ds = random_split(
			train_ds,
			[split_point, len(train_ds) - split_point],
			torch.Generator())

	# DataLoadery
	train_dl = DataLoader(train_ds, batch_size, shuffle = True)
	valid_dl = DataLoader(valid_ds, batch_size, shuffle = True)
	
	return train_dl, valid_dl, inv_map


if __name__ == '__main__':
	main()
