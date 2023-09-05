import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import reduce


model = ['DLA_manual','ResNet18']
batch = ['16','32','64','128']
lr = ['0.001','0.005','0.010']

def main():
	plt.style.use('default')

	var = ['model','batch','lr'][0]

	plot_by(var)
	plt.show()



def frame_data(filename, extra = ''):
	
	data = []
	data_dict = dict()

	with open(join('output data', filename), 'r') as file:
		for line in file:
			data.append(line.replace(',','.').split('	'))

	heads = data.pop(0)[:6]

	heads[1:] = [head + extra for head in heads[1:]]	

	for i in range(len(heads)):
		data_dict[heads[i]] = [float(row[i])  if i != 0 else int(row[i]) for row in data]
	
	df = pd.DataFrame(data_dict).set_index('epoch')

	df['valid loss mean' + extra] = df['valid loss' + extra].rolling(5).mean()
	df['valid loss std' + extra] = df['valid loss' + extra].rolling(5).std()
	df['train loss mean' + extra] = df['train loss' + extra].rolling(5).mean()
	df['train loss std' + extra] = df['train loss' + extra].rolling(5).std()

	return df





def plot_by(var, m_idx = 0, b_idx = 0, lr_idx = 0, valid = True):
	frames = []
	prefix = 'valid' if valid else 'train'

	match var:
		case 'model':
			frames = [frame_data(f'{m} batch{batch[b_idx]} lr{lr[lr_idx]} epoch120.txt', ' ' + m) for m in model]
		case 'batch':
			frames = [frame_data(f'{model[m_idx]} batch{b} lr{lr[lr_idx]} epoch120.txt', ' ' + b) for b in batch]
		case 'lr':
			frames = [frame_data(f'{model[m_idx]} batch{batch[b_idx]} lr{rate} epoch120.txt', ' ' + rate) for rate in lr]

	df = reduce(
		lambda left,right:
		pd.merge(left,right,on=['epoch'],how='outer'),
		frames
		)


	match var:
			case 'model':
				df[[prefix + ' loss mean ' + m for m in model]].plot(figsize=(16,8), title=f'plot by model        batch = {batch[b_idx]}        lr = {lr[lr_idx]}')
			case 'batch':
				df[[prefix + ' loss mean ' + b for b in batch]].plot(figsize=(16,8), title=f'plot by batch        {model[m_idx]}        lr = {lr[lr_idx]}')
			case 'lr':
				df[[prefix + ' loss mean ' + rate for rate in lr]].plot(figsize=(16,8), title=f'plot by lr        {model[m_idx]}        batch = {batch[b_idx]}')
	






if __name__ == '__main__':
	main()