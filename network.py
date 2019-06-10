import torch
import torch.nn as nn
from torch.autograd import Variable

### https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
class ConvNet(nn.Module):
	def __init__(self, nClasses, imageSize, nConvLayers=3, nchannels=32):
		super(ConvNet, self).__init__()
		print("[Network] ", end="")

		self.convLayers = []

		### CONVOLUTIONAL ###
		for iConvLayer in range(nConvLayers):
			in_channels = 1 if iConvLayer == 0 else nchannels
			layer = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=nchannels, kernel_size=5, stride=1, padding=2),
				nn.ReLU(),
				nn.BatchNorm2d(nchannels),
				nn.MaxPool2d(kernel_size=2, stride=2))
			self.convLayers.append(layer)
			print("%dx%dx%d -> " % (layer[0].in_channels, imageSize, imageSize), end="")
			imageSize //= 2
			print("%dx%dx%d|" % (layer[0].out_channels, imageSize, imageSize), end="")

		self.convLayers = nn.ModuleList(self.convLayers)

		### ADAPTIVE ###
		self.adaptive = nn.AdaptiveAvgPool2d((8, 8))
		print("Adaptive -> %dx%dx%d" % (self.convLayers[-1][0].out_channels, 8, 8), end="")

		### FULLY CONNECTED ###
		self.drop_out = nn.Dropout()
		print("|dropout|", end="")

		self.fc = nn.Linear(in_features=nchannels*8*8, out_features=nClasses)
		print("1x%d -> 1x%d" % (self.fc.in_features, self.fc.out_features))

	def forward(self, x):
		out = x
		for layer in self.convLayers:
			out = layer(out)

		out = self.adaptive(out)
		out = out.reshape(out.size(0), -1)
		
		out = self.drop_out(out)
		out = self.fc(out)
		return out
