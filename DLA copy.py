# przedstawić klasyfikację, nie generację
# sygnały ekg (szeregi czasowe)
# detekcja cech
# sieci splotowe


from data_prep import loader

import math

import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
#import gc
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
import matplotlib.pyplot as plt

BatchNorm = nn.BatchNorm1d
Conv = nn.Conv1d

def main():
	train_dl, _ = loader(batch_size=50)

	model = DLA_manual()

	for rec, lab in train_dl:
		result = model(rec)

		print('\nalleluja!\n')
		print(result.shape)
		break

	# blocks_1 = nn.ModuleList([
	# 	BasicBlock(1,8,2),
	# 	BasicBlock(8,8)])

	# blocks_2 = nn.ModuleList([
	# 	BasicBlock(8,16,2),
	# 	BasicBlock(16,16),
	# 	BasicBlock(16,16),
	# 	BasicBlock(16,16)])


	# aggr = Aggr_block(16,8, kernel_size=1, residual=False)# sekcja/drzewo 1
	
	# aggr_2 = nn.ModuleList([
	# 	Aggr_block(32,16, kernel_size=1, residual=False),# sekcja/drzewo 2
	# 	Aggr_block(8+3*16,16, kernel_size=1, residual=False)])
 
 
	
	# downsample = nn.MaxPool1d(1,2)
	# projection = Conv(1,8,1)
	# projection2 = Conv(8,16,1)
 
	# for rec, lab in train_dl:
	# 	b1 = blocks_1[0](rec, projection(downsample(rec)))
	# 	b2 = blocks_1[1](b1)
	# 	a1 = aggr(b1,b2)
		
	# 	# print('projekcja jest ok')
	# 	down = downsample(a1)
	# 	# print(res.shape)
	# 	res = projection2(down)
		

	# 	print(a1.shape)
	# 	# print(res.shape)
	# 	b3 = blocks_2[0](a1, res)
	# 	b4 = blocks_2[1](b3)
	# 	a2 = aggr_2[0](b3,b4)
	# 	b5 = blocks_2[2](a2)
	# 	b6 = blocks_2[3](b5)

	# 	# print(b6.shape)
	# 	a3 = aggr_2[1](down,a2,b5,b6)
		
	# 	break


class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, dilation=1):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.conv1 = Conv(in_channels, out_channels, kernel_size=3,
							stride=stride, padding=dilation,
							bias=False, dilation=dilation)
		self.bn1 = BatchNorm(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = Conv(out_channels, out_channels, kernel_size=3,
							stride=1, padding=dilation,
							bias=False, dilation=dilation)
		self.bn2 = BatchNorm(out_channels)
		


	def forward(self, x, residual = None):
		residual = x if residual is None else residual

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)


		out += residual
		out = self.relu(out)

		return out



class Aggr_block(nn.Module):
	"""
	residual - wzór (4)
	res = true -> wyróżnia x[0], czyli wyjście poprzedniego aggr_block'a
	"""
	def __init__(self, in_channels, out_channels, kernel_size, residual=False):
		super().__init__()
		self.conv = Conv(
			in_channels, out_channels, kernel_size,
			stride=1, bias=False, padding=(kernel_size - 1) // 2)
		self.bn = BatchNorm(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.residual = residual

	def forward(self, *x):
#		children = x
		x = self.conv(torch.cat(x, 1))
		x = self.bn(x)

#		projekcja
#		if self.residual:
#			for i in children:
#				print(i.shape)
#				break
#			print(x.shape)
#			nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False)
#			x += children[0]
		x = self.relu(x)

		return x








class DLA_manual(nn.Module):
	def __init__(self, chan = [1,8,16,32,64,128], block = BasicBlock) -> None:
		super().__init__()
		
		self.base_layer = nn.Sequential(
			Conv(chan[0], chan[1], kernel_size=7, stride=1,
					  padding=3, bias=False),
			BatchNorm(chan[1]),
			nn.ReLU(inplace=True))

		self.blocks_1 = nn.ModuleList([
			block(chan[1],chan[2],2),
			block(chan[2],chan[2])])

		self.blocks_2 = nn.ModuleList([
			block(chan[2],chan[3],2),
			block(chan[3],chan[3]),
			block(chan[3],chan[3]),
			block(chan[3],chan[3])])

		self.blocks_3 = nn.ModuleList([
			block(chan[3],chan[4],2),
			block(chan[4],chan[4]),
			block(chan[4],chan[4]),
			block(chan[4],chan[4]),
			block(chan[4],chan[4]),
			block(chan[4],chan[4]),
			block(chan[4],chan[4]),
			block(chan[4],chan[4])])

		self.blocks_4 = nn.ModuleList([
			block(chan[4],chan[5],2),
			block(chan[5],chan[5])])

		self.aggr_1 = Aggr_block(2*chan[2],chan[2], kernel_size=1, residual=False)

		self.aggr_2 = nn.ModuleList([
			Aggr_block(2*chan[3],chan[3], kernel_size=1, residual=False),
			Aggr_block(chan[2]+3*chan[3],chan[3], kernel_size=1, residual=False)])
		
		self.aggr_3 = nn.ModuleList([
			Aggr_block(2*chan[4],chan[4], kernel_size=1, residual=False),
			Aggr_block(3*chan[4],chan[4], kernel_size=1, residual=False),
			Aggr_block(2*chan[4],chan[4], kernel_size=1, residual=False),
			Aggr_block(4*chan[4]+chan[3],chan[4], kernel_size=1, residual=False)])

		self.aggr_4 = Aggr_block(2*chan[5] + chan[4],chan[2], kernel_size=1, residual=False)
  
		self.downsample = nn.MaxPool1d(1,2)
		self.projection1 = Conv(chan[1],chan[2],1)
		self.projection2 = Conv(chan[2],chan[3],1)
		self.projection3 = Conv(chan[3],chan[4],1)
		self.projection4 = Conv(chan[4],chan[5],1)


	def forward(self, x):
		x = self.base_layer(x)

		# sekcja 1
		res = self.projection1(self.downsample(x))
		b1 = self.blocks_1[0](x, res)
		b2 = self.blocks_1[1](b1)
		a1 = self.aggr_1(b1,b2)


		# sekcja 2
		# print('projekcja jest ok')
		down = self.downsample(a1)
		# print(res.shape)
		res = self.projection2(down)

		# print(a1.shape)
		# print(res.shape)
		b3 = self.blocks_2[0](a1, res)
		b4 = self.blocks_2[1](b3)
		a2 = self.aggr_2[0](b3,b4)
		b5 = self.blocks_2[2](a2)
		b6 = self.blocks_2[3](b5)
		a3 = self.aggr_2[1](down,a2,b5,b6)
		# print(b5.shape)
		# print(b6.shape)



		# sekcja 3
		down = self.downsample(a3)
		# print(res.shape)
		res = self.projection3(down)

		b7 = self.blocks_3[0](a3,res)
		b8 = self.blocks_3[1](b7)
		a4 = self.aggr_3[0](b7,b8)

		b9 = self.blocks_3[2](a4)
		b10 = self.blocks_3[3](b9)
		a5 = self.aggr_3[1](a4, b9, b10)

		b11 = self.blocks_3[4](a5)
		b12 = self.blocks_3[5](b11)
		a6 = self.aggr_3[2](b11,b12)

		b13 = self.blocks_3[6](a6)
		b14 = self.blocks_3[7](b13)
		a7 = self.aggr_3[3](down, a5, a6, b13, b14)
  
		# sekcja 4
		down = self.downsample(a7)
		res = self.projection4(down)

		b15 = self.blocks_4[0](a7,res)
		b16 = self.blocks_4[1](b15)
		out = self.aggr_4(down, b15, b16)

		return out



if __name__ == '__main__':
	main()