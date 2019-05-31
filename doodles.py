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
from network import ConvNet
from shutil import copyfile

print("\n\nDoodle CNN")
print("Programmed for pytorch v1.0.0")
torch.cuda.empty_cache()

# Image.fromarray(imData * 255).resize((1000, 1000)).show()

#####################################################################################
#####################################################################################
#####################################################################################

NCLASSES = 10
NFILES = 80
IMAGE_SIZE = 128
EPOCHS = 20
NBATCHES = NCLASSES * NFILES * 0.1

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Load data ###
data, iClasses, classToLabel = None, None, None
if True:
	print("Loading and preprocessing images..")
	data, iClasses, classToLabel = dataloader.loadAndPrepAllImages(NCLASSES, NFILES, IMAGE_SIZE)
	if False:
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
modelDir = "model_%s_%d_%d_%d" % (nowStr, IMAGE_SIZE, NCLASSES, NFILES)
os.mkdir(modelDir)
copyfile("./network.py", modelDir + "/network.py")

### Profile storage ###
dataInGb   = sum([d.nbytes for d in data]) / 1e9
trainInGb  = sys.getsizeof(dataTrain.storage()) / (1e9)
testInGb   = sys.getsizeof(dataTest.storage()) / 1e9

print("   Device : %s" % device)
print("Directory : %s" % modelDir)
print("   Memory : data=%0.2fGB train=%0.2fGB test=%0.2fGB" % (dataInGb, trainInGb, testInGb))
print(" Training : NTRAIN=%d NTEST=%d EPOCHS=%d NBATCHES=%d batchSize=%d" % (NTRAIN, NTEST, EPOCHS, NBATCHES, batchSize))
print("")

################
### Training ###
################
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
correct = 0
for nB in range(0, NTEST, batchSize):
	y = model(dataTest[nB:nB+batchSize])
	_, predicted = torch.max(y.data, 1)
	correct += (predicted == labelsTest[nB:nB+batchSize]).sum().item()
print("Accuracy : %0.2f" % (correct / NTEST))


correct = [0] * NCLASSES
data = torch.FloatTensor(data)
for i in range(0, NCLASSES * NFILES):
	d = data[i]
	c = iClasses[i]
	l = classToLabel[c]
	
	output = model(data[i:i+1])
	_, predicted = torch.max(output.data, 1)
	p = predicted.item()
	if c == p:
		correct[c] += 1
	print("\r  %s - %d" % (l, correct[c]), " "*20, end="", flush=True)
print("\r", " "*40)

performance = list(zip(correct, classToLabel))
performance.sort(reverse=True)
for i, c in performance:
	print(c.rjust(20), "%0.2f" % (i/NFILES))
