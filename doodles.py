import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
import dataloader
import sys
import math
import pickle

print("\n\nDoodle CNN")
print("Programmed for pytorch v1.0.0")

# Image.fromarray(imData * 255).resize((1000, 1000)).show()

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

#####################################################################################
#####################################################################################
#####################################################################################

NCLASSES = 250
NFILES = 80
IMAGE_SIZE = 128
EPOCHS = 20
NBATCHES = NCLASSES * NFILES * 0.01

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### Load data ###
data = None
iClasses = None
classToLabel = None
if False:
	print("Loading and preprocessing images..")
	data, iClasses, classToLabel = dataloader.loadAndPrepAllImages(NCLASSES, NFILES, IMAGE_SIZE, rootfolder="/home/emiel/sketches_png")
	print("Storing files as pickles..")
	with open("data.pickle", 'wb') as pickle_file:
		pickle.dump(data, pickle_file, pickle.HIGHEST_PROTOCOL)
	with open("iClasses.pickle", 'wb') as pickle_file:
		pickle.dump(iClasses, pickle_file, pickle.HIGHEST_PROTOCOL)
	with open("classToLabel.pickle", 'wb') as pickle_file:
		pickle.dump(classToLabel, pickle_file, pickle.HIGHEST_PROTOCOL)
else:
	print("Loading pickles..")
	with open("data.pickle", "rb") as pickle_file:
	   data = pickle.load(pickle_file)
	with open("iClasses.pickle", "rb") as pickle_file:
	   iClasses = pickle.load(pickle_file)
	with open("classToLabel.pickle", "rb") as pickle_file:
	   classToLabel = pickle.load(pickle_file)


print("Converting data to Tensors and moving data to device.. ")
### Split data into test and train ###
NTRAIN = int(NCLASSES * NFILES * 0.8)
NTEST = NCLASSES * NFILES - NTRAIN
# Create random indices
indicesTrain = list(np.random.choice(NCLASSES * NFILES, NTRAIN, replace=False))
indicesTest  = [i for i in range(0, NCLASSES * NFILES) if i not in indicesTrain]
# Train sets, move to device
dataTrain   = torch.FloatTensor([data[i]     for i in indicesTrain]).to(device)
labelsTrain = torch.FloatTensor([iClasses[i] for i in indicesTrain]).long().to(device)
# Test sets, keep on cpu
dataTest   = torch.FloatTensor([data[i]     for i in indicesTest])
labelsTest = torch.FloatTensor([iClasses[i] for i in indicesTest]).long()
## .long() fixes : RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target'

### Create model ###
model = ConvNet(NCLASSES, IMAGE_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batchSize = math.ceil(NTRAIN / NBATCHES)

### Create directory to store model weights ###
nowStr = datetime.now().strftime("%y%m%d-%H%M%S")
modelDir = "./model_%s_%dx%d_%d_%d" % (nowStr, IMAGE_SIZE, IMAGE_SIZE, NCLASSES, NFILES)
os.mkdir(modelDir)

### Profile storage ###
dataInGb   = sum([d.nbytes for d in data]) / 1e9
trainInGb  = sys.getsizeof(dataTrain.storage()) / (1e9)
testInGb   = sys.getsizeof(dataTest.storage()) / 1e9

print("   Device : %s" % device)
print("Directory : %s" % modelDir)
print("   Memory : data=%0.2fGB train=%0.2fGB test=%0.2fGB" % (dataInGb, trainInGb, testInGb))
print(" Training : NTRAIN=%d NTEST=%d EPOCHS=%d NBATCHES=%d batchSize=%d" % (NTRAIN, NTEST, EPOCHS, NBATCHES, batchSize))
print("")

for i in range(0, EPOCHS):
	
	### Train batch
	for nB in range(0, NTRAIN, batchSize):
		print("\r  Batching %d/%d" % (nB+batchSize, NTRAIN), end=" "*20, flush=True)
		
		# Run the forward pass
		y = model(dataTrain[nB:nB+batchSize])
		loss = criterion(y, labelsTrain[nB:nB+batchSize])
		
		# Backprop and perform Adam optimisation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	### Calculate accuracy
	acc = 0.0
	for nB in range(0, NTRAIN, batchSize):
		print("\r  Accuracy %d/%d" % (nB+batchSize, NTRAIN), end=" "*20, flush=True)

		# Forward pass
		y = model(dataTrain[nB:nB+batchSize])
		_, predicted = torch.max(y.data, 1)
		correct = (predicted == labelsTrain[nB:nB+batchSize]).sum().item()
		acc += correct / batchSize
	acc = acc / NBATCHES	

	print("\r  epoch=%d" % i, "accuracy=%0.2f" % acc)
	
	### Store current model
	torch.save(model.state_dict(), modelDir + "/%d_%0.2f.model" % (i, acc))


### Calculate accuracy
print("\nTesting CNN on cpu..")
model = model.to("cpu")
acc = 0.0
for nB in range(0, NTEST, batchSize):
	print("\r  Accuracy %d/%d" % (nB+batchSize, NTRAIN), end=" "*20, flush=True)
	# Forward pass
	y = model(dataTest[nB:nB+batchSize])
	_, predicted = torch.max(y.data, 1)
	correct = (predicted == labelsTest[nB:nB+batchSize]).sum().item()
	acc += correct / batchSize
acc = acc / NBATCHES
print("Accuracy : %0.2f" % (correct / NTEST))