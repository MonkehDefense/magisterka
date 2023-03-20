import data_prep
import DLA

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn




def main():
	lr = 5e-3

	batch_size=128
	loss_fn = nn.CrossEntropyLoss()

	TDL, VDL, _ = data_prep.loader(batch_size=batch_size, split_point=.9)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	TDL = DeviceDataLoader(TDL, device)
	VDL = DeviceDataLoader(VDL, device)



	# model = nn.Sequential(
	# 	nn.Conv1d(1,8,3, padding = 1),
	# 	nn.MaxPool1d(2),
	# 	nn.ReLU(),
	# 	nn.Conv1d(8,16,3, padding = 1),
	# 	nn.MaxPool1d(2),
	# 	nn.ReLU(),
	# 	nn.Conv1d(16,32,3, padding = 1),
	# 	nn.MaxPool1d(2),
	# 	nn.ReLU(),
	# 	nn.Conv1d(32,64,3, padding = 1),
	# 	nn.MaxPool1d(2),
	# 	nn.ReLU(),
	# 	nn.Conv1d(64,128,3, padding = 1),
	# 	nn.MaxPool1d(2),
	# 	nn.ReLU(),
	# 	nn.Conv1d(128,256,3, padding = 1),
	# 	nn.MaxPool1d(2),
	# 	nn.ReLU(),
	# 	nn.Flatten(),
	# 	nn.Linear(32*256, 256),
	# 	nn.ReLU(),
	# 	nn.Linear(256,4)
	# ).to(device)


	model = DLA.DLA_manual().to(device)


	# optimizer = torch.optim.SGD(model.parameters(),
	# 							lr,
	# 							momentum,
	# 							weight_decay = weight_decay)
 
 
	optimizer = torch.optim.Adam(model.parameters(),lr)



	losses, accuracies = fit(60, model, loss_fn, optimizer, TDL, VDL, accuracy)

	

	# print(losses)
	# print(accuracies)

	# p = plt.plot(losses)
	# plt.show(p)















def accuracy(outputs, labels):
	_, preds = torch.max(outputs, dim=1)
	return torch.sum(preds == labels).item() / len(preds)

def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
	preds = model(xb)
	loss = loss_func(preds, yb)

	if opt is not None:
		loss.backward()
		opt.step()
		opt.zero_grad()

	metric_result = None
	if metric is not None:
		metric_result = metric(preds, yb)

	return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, valid_dl, metric=None):
	with torch.no_grad():
		results = [loss_batch(model,loss_fn,xb,yb,metric=metric) for xb,yb in valid_dl]

		losses, nums, metrics = zip(*results)

		total = np.sum(nums)
		avg_loss = np.sum(np.multiply(losses, nums)) / total
		avg_metric = None
		if metric is not None:
			avg_metric = np.sum(np.multiply(metrics, nums)) / total

		return avg_loss, total, avg_metric


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
	losses = []
	metrics = []
	for epoch in range(epochs):
		for xb, yb in train_dl:
			loss, _, _ = loss_batch(model,loss_fn,xb,yb,opt)

		result = evaluate(model, loss_fn, valid_dl,metric)
		val_loss, total, val_metric = result

		losses.append(val_loss)

		if metric is None:
			print('Epoch [{}/{}], Train_Loss: {:.4f}, Val_Loss: {:.4f}'.format(epoch+1, epochs, loss, val_loss))
		else:
			metrics.append(val_metric)
			print('Epoch [{}/{}], Train_Loss: {:.4f}, Val_Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, loss, val_loss, metric.__name__, val_metric))

	return losses, metrics













def to_device(data, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking = True)

class DeviceDataLoader():
	def __init__(self, dl, device):
		self.dl = dl
		self.device = device

	def __iter__(self):
		for b in self.dl:
			yield to_device(b, self.device)

	def __len__(self):
		return len(self.dl)



if __name__ == '__main__':
	main()