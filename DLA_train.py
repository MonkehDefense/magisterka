import data_prep
import DLA_copy

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np




def main():
	lr = .001
	TDL, VDL, _ = data_prep.loader(batch_size=50)

	model = DLA_copy.DLA_manual()


	# for rec, lab in TDL:
	# 	print(list(model.parameters()))
	# 	# model(rec)
	# 	# print(lab)
	# 	# result = model(rec)

	# 	# print('\nalleluja!\n')
	# 	# print(result.shape)
	# 	break


	loss_fn = F.cross_entropy


	for rec, lab in TDL:
		outputs = model(rec)
		# loss = loss_fn(outputs, lab)
		# torch.optim.Adam(model.parameters(), lr=lr)


		val_loss, total, val_acc = evaluate(model, loss_fn, VDL, accuracy)
		print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss,val_acc))

		# probs = F.softmax(outputs, dim=1)

		# print(outputs)
		# print(probs)

		# max_probs, preds = torch.max(probs,1)
		# print(preds)
		# print(lab == preds)
		# print(accuracy(lab, preds))
		
		
		# print(probs.dtype)
		# print(lab.dtype)
		# loss = loss_fn(outputs, lab)
		# loss = loss_fn(probs, lab)
		
		# print(loss.item())
		# print(np.e ** ( - loss.item()))


		
		break
















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
	for epoch in range(epochs):
		for xb, yb in train_dl:
			loss, _, _ = loss_batch(model,loss_fn,xb,yb,opt)

		result = evaluate(model, loss_fn, valid_dl,metric)
		val_loss, total, val_metric = result

		if metric is None:
			print('Epoch [{}/{}], Loss: {:.4f}'
				.format(epoch+1, epochs, val_loss))
		else:
			print('Epoch [{}/{}], Loss: {:.4f}, {}: {:,4f}'
				.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

	pass


if __name__ == '__main__':
	main()