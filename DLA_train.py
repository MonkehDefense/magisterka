import data_prep
import DLA

import torch
# import torch.nn.functional as F
# from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle




def main():
	lr = 5e-3

	batch_size=128
	loss_fn = nn.CrossEntropyLoss()

	TDL, VDL, _ = data_prep.loader(batch_size=batch_size, split_point=.9)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	TDL_CUDA = DeviceDataLoader(TDL, device)
	VDL_CUDA = DeviceDataLoader(VDL, device)



	# model = DLA.SimpleSeq().to(device)
	# model = DLA.SimpleResNet().to(device)
	model = DLA.ResNet18().to(device)
	# model = DLA.DLA_manual().to(device)


	optimizer = torch.optim.Adam(model.parameters(),lr)

	losses, accuracies = fit(50, model, loss_fn, optimizer, TDL_CUDA, VDL_CUDA, accuracy)

	cm = my_cm(VDL,model,4,device)
	print(cm)
	fpr, tpr, roc_auc = my_roc_curve(VDL, model, 4, device)
	plot_roc_curve(fpr, tpr, roc_auc, 4, type(model).__name__)













def my_cm(dataloader, model, classes, device):
	cm = np.zeros((classes,classes))
	with torch.no_grad():
		for x, y in dataloader:
			x = x.to(device)
			y_pred = model(x).argmax(dim=1).cpu()
			cm += confusion_matrix(y, y_pred, labels = list(range(classes)))

	return cm.astype(int)



def my_roc_curve(dataloader, model, classes, device):
	y_score = np.zeros((0, classes))
	y_true = np.zeros((0, classes))
	with torch.no_grad():
		for x, y in dataloader:
			x = x.to(device)
			y_score = np.concatenate((y_score, model(x).cpu()), axis=0)
			y_true = np.concatenate((y_true, label_binarize(y, classes=list(range(classes)))), axis=0)
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(classes):
		fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc, classes, model_name):
	plt.figure()
	lw = 2
	plt.plot(fpr["micro"], tpr["micro"],
			 label='micro-average(area = {0:0.2f})'
				   ''.format(roc_auc["micro"]),
			 color='deeppink', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw,
				 label='class {0} (area = {1:0.2f})'
				 ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.title(model_name)
	plt.show()








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