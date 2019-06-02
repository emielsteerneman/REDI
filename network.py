import torch
import torch.nn as nn
from torch.autograd import Variable

### https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
class ConvNet(nn.Module):
	def __init__(self, nClasses, imageSize):
		super(ConvNet, self).__init__()
		print("[Network] ", end="")
		
		### CONVOLUTIONAL ###
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		print("%dx%dx%d -> " % (self.layer1[0].in_channels, imageSize, imageSize), end="")
		imageSize = imageSize // 2
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		print("%dx%dx%d -> " % (self.layer2[0].in_channels, imageSize, imageSize), end="")
		imageSize = imageSize // 2

		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.ReLU())
		print("%dx%dx%d -> " % (self.layer3[0].in_channels, imageSize, imageSize), end="")

		self.layer4 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		print("%dx%dx%d -> " % (self.layer4[0].in_channels, imageSize, imageSize), end="")
		imageSize = imageSize // 2

		self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		print("%dx%dx%d -> " % (self.layer5[0].in_channels, imageSize, imageSize), end="")
		imageSize = imageSize // 2
		print("%dx%dx%d -> " % (self.layer5[0].out_channels, imageSize, imageSize), end="")
		
		self.convLayers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]

		self.adaptive = torch.nn.AdaptiveAvgPool2d((8, 8))
		print("adaptive %d -> " % (self.layer5[0].out_channels), end="")

		### FULLY CONNECTED ###
		self.drop_out = nn.Dropout()
		print("dropout -> ", end="")

		self.fc1 = nn.Linear(in_features=4096, out_features=512)
		print("1x%d -> " % (self.fc1.in_features), end="")

		self.fc2 = nn.Linear(in_features=512, out_features=nClasses)
		print("1x%d -> 1x%d" % (self.fc2.in_features, self.fc2.out_features))

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.adaptive(out)
		out = out.reshape(out.size(0), -1)
		
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out
