import torch
import torch.nn as nn
from torch.autograd import Variable

### https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
class ConvNet(nn.Module):
	def __init__(self, nClasses, imageSize):
		super(ConvNet, self).__init__()
		# 1 128x128 image goes in, 32 64x64 images come out
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		imageSize = imageSize // 2

		# 32 64x64 images go in, 64 32x32 image comes out
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		imageSize = imageSize // 2

		# 32 64x64 images go in, 64 32x32 image comes out
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		imageSize = imageSize // 2

		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(in_features=imageSize * imageSize * 32, out_features=(imageSize//4) * (imageSize//4) * 64)
		self.fc2 = nn.Linear(in_features=(imageSize//4) * (imageSize//4) * 64, out_features=nClasses)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.drop_out(out)
		out = self.layer3(out)
		out = out.reshape(out.size(0), -1)
		
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out
