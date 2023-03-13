from data_prep import loader

# import math

import torch
# import torchvision
import torch.nn as nn
# import torch.nn.functional as F
#import gc
# from torch.utils.data import TensorDataset, DataLoader
# import torchvision.transforms as tt
from torch.utils.data import random_split
# import matplotlib.pyplot as plt

BatchNorm = nn.BatchNorm1d
Conv = nn.Conv1d

def main():
	train_dl, valid_dl,_ = loader(batch_size=50)

	model = DLA_manual()

	for rec, lab in train_dl:
		result = model(rec)

		# print('\nalleluja!\n')
		# print(result.shape)
		break









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
	# pool_size = 7 256
	def __init__(self, chan = [1,8,16,32,64,128,256], block = BasicBlock, pool_size = 64, num_classes = 4) -> None:
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
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5]),
			block(chan[5],chan[5])])




		
		self.blocks_5 = nn.ModuleList([
			block(chan[5],chan[6],2),
			block(chan[6],chan[6])])




		self.aggr_1 = Aggr_block(2*chan[2],chan[2], kernel_size=1, residual=False)

		self.aggr_2 = nn.ModuleList([
			Aggr_block(2*chan[3],chan[3], kernel_size=1, residual=False),
			Aggr_block(chan[2]+3*chan[3],chan[3], kernel_size=1, residual=False)])
		
		self.aggr_3 = nn.ModuleList([
			Aggr_block(2*chan[4],chan[4], kernel_size=1, residual=False),
			Aggr_block(3*chan[4],chan[4], kernel_size=1, residual=False),
			Aggr_block(2*chan[4],chan[4], kernel_size=1, residual=False),
			Aggr_block(4*chan[4]+chan[3],chan[4], kernel_size=1, residual=False)])
		

		self.aggr_4 = nn.ModuleList([
			Aggr_block(2*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(3*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(2*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(4*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(2*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(3*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(2*chan[5],chan[5], kernel_size=1, residual=False),
			Aggr_block(5*chan[5]+chan[4],chan[5], kernel_size=1, residual=False)])


		self.aggr_5 = Aggr_block(2*chan[6] + chan[5],chan[6], kernel_size=1, residual=False)

		self.downsample = nn.MaxPool1d(1,2)
		self.projection1 = Conv(chan[1],chan[2],1)
		self.projection2 = Conv(chan[2],chan[3],1)
		self.projection3 = Conv(chan[3],chan[4],1)
		self.projection4 = Conv(chan[4],chan[5],1)
		self.projection5 = Conv(chan[5],chan[6],1)

		
		
		self.avgpool = nn.AvgPool1d(pool_size)
		self.fc = Conv(chan[6], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

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
		a8 = self.aggr_4[0](b15,b16)

		b17 = self.blocks_4[2](a8)
		b18 = self.blocks_4[3](b17)
		a9 = self.aggr_4[1](a8, b17, b18)

		b19 = self.blocks_4[4](a9)
		b20 = self.blocks_4[5](b19)
		a10 = self.aggr_4[2](b19,b20)

		b21 = self.blocks_4[6](a10)
		b22 = self.blocks_4[7](b21)
		a11 = self.aggr_4[3](a9, a10, b21, b22)

		


		b23 = self.blocks_4[8](a11)
		b24 = self.blocks_4[9](b23)
		a12 = self.aggr_4[4](b23,b24)

		b25 = self.blocks_4[10](a12)
		b26 = self.blocks_4[11](b25)
		a13 = self.aggr_4[5](a12, b25, b26)

		b27 = self.blocks_4[12](a13)
		b28 = self.blocks_4[13](b27)
		a14 = self.aggr_4[6](b27,b28)

		b29 = self.blocks_4[14](a14)
		b30 = self.blocks_4[15](b29)
		a15 = self.aggr_4[7](down, a11, a13, a14, b29, b30)




		# sekcja 5
		down = self.downsample(a15)
		res = self.projection5(down)

		b31 = self.blocks_5[0](a15,res)
		b32 = self.blocks_5[1](b31)
		out = self.aggr_5(down, b31, b32)






		# końcowa faza
		# print(out.shape)
		out = self.avgpool(out)
		out = self.fc(out)
		out = out.view(out.size(0), -1)
		# print(out.shape)

		return out



if __name__ == '__main__':
	main()