import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import reduce
from os.path import join



w = 5
model = ['DLA_manual','ResNet18']
# batch = ['16','32','64','128']
batch = ['16','128']
lr = ['0.001','0.005','0.010']

cols = ['valid loss', 'train loss', 'valid loss mean', 'train loss mean', 'Cohen kappa', 'F1-score', 'MCC']

def main():
	plt.style.use('default')
	
	# var = ['model','batch','lr'][0]

	output_dir = 'grafy'

	# df = pd.merge(
	# 	frame_data('DLA_manual batch128 lr0.001 epoch120.txt', ' nowa'),
	# 	frame_data('DLA old.txt', ' oryginalna'),
	# 	'outer',
	# 	'epoch'
	# 	)
	
	# df[['train loss nowa','train loss oryginalna']].plot(figsize=(16,8), title='DLA old vs new        batch = 128        lr = 0.001')

	# print(df.head())

	plot_by('batch', m_idx=1)

	plt.grid(True, 'both', 'both')




	# var = 'model'
	# for b in range(4):
	# 	for rate in range(3):
	# 		plot_by(var, b_idx=b, m_idx=0, lr_idx=rate, col=cols[2])
	# 		plt.grid(True, 'both', 'both')
	# 		plt.savefig(join(output_dir, f'ByModel {cols[2]} b{batch[b]} lr{lr[rate]}.png'))
	# 		plot_by(var, b_idx=b, m_idx=0, lr_idx=rate, col=cols[3])
	# 		plt.grid(True, 'both', 'both')
	# 		plt.savefig(join(output_dir, f'ByModel {cols[3]} b{batch[b]} lr{lr[rate]}.png'))

	# var = 'batch'
	# for m in range(2):
	# 	for rate in range(3):
	# 		plot_by(var, b_idx=0, m_idx=m, lr_idx=rate, col=cols[2])
	# 		plt.grid(True, 'both', 'both')
	# 		plt.savefig(join(output_dir, f'ByBatch {cols[2]} m{model[m]} lr{lr[rate]}.png'))
	# 		plot_by(var, b_idx=0, m_idx=m, lr_idx=rate, col=cols[3])
	# 		plt.grid(True, 'both', 'both')
	# 		plt.savefig(join(output_dir, f'ByBatch {cols[3]} m{model[m]} lr{lr[rate]}.png'))

	# var = 'lr'
	# for b in range(4):
	# 	for m in range(2):
	# 		plot_by(var, b_idx=b, m_idx=m, lr_idx=0, col=cols[2])
	# 		plt.grid(True, 'both', 'both')
	# 		plt.savefig(join(output_dir, f'ByLR {cols[2]} m{model[m]} b{batch[b]}.png'))
	# 		plot_by(var, b_idx=b, m_idx=m, lr_idx=0, col=cols[3])
	# 		plt.grid(True, 'both', 'both')
	# 		plt.savefig(join(output_dir, f'ByLR {cols[3]} m{model[m]} b{batch[b]}.png'))
	



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

	df['valid loss mean' + extra] = df['valid loss' + extra].rolling(w).mean()
	df['valid loss std' + extra] = df['valid loss' + extra].rolling(w).std()
	df['train loss mean' + extra] = df['train loss' + extra].rolling(w).mean()
	df['train loss std' + extra] = df['train loss' + extra].rolling(w).std()

	return df





def plot_by(var, m_idx = 0, b_idx = 0, lr_idx = 0, col = cols[2]):
	frames = []

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


	# print(df['valid loss mean 0.001'])

	match var:
			case 'model':
				df[[col + ' ' + m for m in model]].plot(figsize=(16,8), title=f'plot by model        batch = {batch[b_idx]}        lr = {lr[lr_idx]}')
				if col.endswith('mean'):
					for m in model: plt.fill_between(df.index,
									df[col + ' ' + m] - df[col.replace('mean','std') + ' ' + m ],
									df[col + ' ' + m] + df[col.replace('mean','std') + ' ' + m ],
									alpha = .15)
			case 'batch':
				df[[col + ' ' + b for b in batch]].plot(figsize=(16,8), title=f'plot by batch        {model[m_idx]}        lr = {lr[lr_idx]}')
				if col.endswith('mean'):
					for b in batch: plt.fill_between(df.index,
									df[col + ' ' + b] - df[col.replace('mean','std') + ' ' + b ],
									df[col + ' ' + b] + df[col.replace('mean','std') + ' ' + b ],
									alpha = .15)
			case 'lr':
				df[[col + ' ' + rate for rate in lr]].plot(figsize=(16,8), title=f'plot by lr        {model[m_idx]}        batch = {batch[b_idx]}')
				if col.endswith('mean'):
					for rate in lr: plt.fill_between(df.index,
									df[col + ' ' + rate] - df[col.replace('mean','std') + ' ' + rate ],
									df[col + ' ' + rate] + df[col.replace('mean','std') + ' ' + rate ],
									alpha = .15)
	






if __name__ == '__main__':
	main()