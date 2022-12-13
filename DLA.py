from data_prep import loader

import math

import torch
import torchvision
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
	model = Shallow_NN([1,8,16,32,64,128,256])	
	for rec, lab in train_dl:
		model(rec)
		break


class BasicBlock(nn.Module):
	"""
	stride = 2?
	"""
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
		self.stride = stride

		self.downsample = nn.MaxPool1d(stride)
		self.projection = Conv(in_channels,out_channels,1)


	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.stride > 1:
			residual = self.downsample(x)
		else:
			residual = x

		if self.in_channels != self.out_channels:
			residual = self.projection(residual)

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
		children = x
		# cat(x,0) -> łączymy końcami tensory 1d i przeprowadzamy konwolucję 1d na takim tensorze
		# cat(x,1) -> tensory 1d łączymy w tensor 2d wtedy musimy przeprowadzić konwolucję 2d
		# x = self.conv(torch.cat(x, 1))
		x = self.conv(torch.cat(x, 0))
		x = self.bn(x)

		# Przy konwolucji może dojść do podwojenia liczby kanałów
		# children ma dwa razy mniej kanałów. Co z tym zrobić, żeby to dodać?
		if self.residual:
			x += children[0]
		x = self.relu(x)

		return x








class Shallow_NN(nn.Module):
	def __init__(self, channels, num_classes=4, block=BasicBlock, pool_size=7):
		super().__init__()
		self.channels = channels
		self.num_classes = num_classes

		self.base_layer = nn.Sequential(
			Conv(channels[0], channels[1], kernel_size=7, stride=1,
					  padding=3, bias=False),
			BatchNorm(channels[1]),
			nn.ReLU(inplace=True))
			

		self.conv_layer_1 = self._make_conv_level(
			channels[1], channels[1], 1)
		self.conv_layer_2 = self._make_conv_level(
			channels[1], channels[2], 1, stride=1)

		self.blocks_1 = nn.ModuleList([
			block(channels[2],channels[3],2),
			block(channels[3],channels[3])])

		self.blocks_2 = nn.ModuleList([
			block(channels[4],channels[4]),
			block(channels[4],channels[4]),
			block(channels[4],channels[4]),
			block(channels[4],channels[4])])

		self.blocks_3 = nn.ModuleList([
			block(channels[5],channels[5]),
			block(channels[5],channels[5]),
			block(channels[5],channels[5]),
			block(channels[5],channels[5]),
			block(channels[5],channels[5]),
			block(channels[5],channels[5]),
			block(channels[5],channels[5]),
			block(channels[5],channels[5])])

		self.blocks_4 = nn.ModuleList([
			block(channels[6],channels[6]),
			block(channels[6],channels[6])])
		
		self.aggr_nodes = nn.ModuleList([
			Aggr_block(channels[3],channels[4], kernel_size=1, residual=False),# sekcja/drzewo 1
			Aggr_block(channels[4],channels[4], kernel_size=1, residual=False),# sekcja/drzewo 2
			Aggr_block(channels[4],channels[5], kernel_size=1, residual=True),
			Aggr_block(channels[5],channels[5], kernel_size=1, residual=False),# sekcja/drzewo 3
			Aggr_block(channels[5],channels[5], kernel_size=1, residual=False),
			Aggr_block(channels[5],channels[5], kernel_size=1, residual=False),
			Aggr_block(channels[5],channels[6], kernel_size=1, residual=True),
			Aggr_block(channels[6],num_classes, kernel_size=1, residual=True)])# sekcja/drzewo 4

#        self.b1 = block(channels[2],channels[3])
#        self.b2 = block(channels[3],channels[3])
#        self.b3 = block(channels[3],channels[3])
#        self.b4 = block(channels[3],channels[3])
#        self.b5 = block(channels[3],channels[3])
#        self.b6 = block(channels[3],channels[3])
#        self.b7 = block(channels[3],channels[3])
#        self.b8 = block(channels[3],channels[3])
#
#        self.a1 = Aggr_block(channels[3],channels[3], kernel_size=1, residual=False)
#        self.a2 = Aggr_block(channels[3],channels[3], kernel_size=1, residual=False)
#        self.a3 = Aggr_block(channels[3],channels[3], kernel_size=1, residual=False)
#        self.a4 = Aggr_block(channels[3],channels[4], kernel_size=1, residual=False)

		self.avgpool = nn.AvgPool1d(pool_size)
		self.conv_final = Conv(channels[4], num_classes, kernel_size=1,
							stride=1, padding=0, bias=True)

		for m in self.modules():
			if isinstance(m, Conv):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, BatchNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
		modules = []
		for i in range(convs):
			modules.extend([
				Conv(inplanes, planes, kernel_size=3,
						  stride=stride if i == 0 else 1,
						  padding=dilation, bias=False, dilation=dilation),
				BatchNorm(planes),
				nn.ReLU(inplace=True)])
			inplanes = planes
		return nn.Sequential(*modules)

#    def _make_block_chain(self, inplanes, planes, blocks, stride=1, dilation=1):
#        modules = []
#        for i in range(blocks):
#            modules.extend([
#                Conv(inplanes, planes, kernel_size=3,
#                          stride=stride if i == 0 else 1,
#                          padding=dilation, bias=False, dilation=dilation),
#                BatchNorm(planes),
#                nn.ReLU(inplace=True)])
#            inplanes = planes
#        return nn.Sequential(*modules)

	def forward(self, x):
		
		x = self.base_layer(x)
		x = self.conv_layer_1(x)
		x = self.conv_layer_2(x)

		# sekcja/drzewo 1
		b1_1 = self.blocks_1[0](x)
		b1_2 = self.blocks_1[1](b1_1)
		a1_ = self.aggr_nodes[0](b1_1,b1_2)

		# sekcja/drzewo 2
		b2_1 = self.blocks_2[0](a1_)
		b2_2 = self.blocks_2[1](b2_1)
		a2_1 = self.aggr_nodes[1](b2_1,b2_2)

		b2_3 = self.blocks_2[2](a2_1)
		b2_4 = self.blocks_2[3](b2_3)
		print("a1 size =", a1_.size(), "a2.1 size =", a2_1.size(), "b2.3, b2.4 size =", b2_3.size(), b2_4.size())
#        print(torch.cat((b2_1,b2_2),0).size())
#        print("Agregacja 3")
		a2_2 = self.aggr_nodes[2](a1_,a2_1,b2_3,b2_4)
		#The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 1
#        print("Agregacja 3")
		
		# sekcja/drzewo 3
		b3_1 = self.blocks_3[0](a2_2)
		b3_2 = self.blocks_3[1](b3_1)
		a3_1 = self.aggr_nodes[3](b3_1,b3_2)

		b3_3 = self.blocks_3[2](a3_1)
		b3_4 = self.blocks_3[3](b3_3)
		a3_2 = self.aggr_nodes[4](a3_1,b3_3,b3_4)

		b3_5 = self.blocks_3[4](a3_2)
		b3_6 = self.blocks_3[5](b3_5)
		a3_3 = self.aggr_nodes[5](b3_5,b3_6)

		b3_7 = self.blocks_3[6](a3_3)
		b3_8 = self.blocks_3[7](b3_7)
		a3_4 = self.aggr_nodes[6](a2_2,a3_2,a3_3,b3_7,b3_8)

		# sekcja/drzewo 4
		b4_1 = self.blocks_4[0](a3_4)
		b4_2 = self.blocks_4[1](b4_1)
		out = self.aggr_nodes[0](a3_4,b4_1,b4_2)

		out = self.avgpool(out)
		out = self.conv_final(out)
		out = out.view(out.size(0), -1)

		return out



if __name__ == '__main__':
	main()