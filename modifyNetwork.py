import sys
import numpy as np
import dataloader
from torchsummary import summary
import cv2
import torch

# Numpy pretty print
np.set_printoptions(precision=2)

### Load model
modelDir = None
if modelDir == None:
	if len(sys.argv) == 1:
		model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
	else:
		model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(sys.argv[1])
else:
	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(modelDir)

# Set model to evaluation mode (disables dropout layers)
model.eval()
summary(model, (1, IMAGE_SIZE, IMAGE_SIZE), 1, "cpu")

# Loading all data
data, iClasses, classToLabel = dataloader.loadAndPrepAllImages(nFiles=NFILES, imsize=IMAGE_SIZE, classes=CLASSES)
data = torch.FloatTensor(data)
iClasses = np.array(iClasses)

def testPerformance(model, data, iClasses):
	pCorrect= 0
	table   = np.zeros((NCLASSES, NCLASSES))

	for i in range(len(data)):
		c = iClasses[i]
		output = model(data[i:i+1])
		_, predicted = torch.max(output.data, 1)
		p = predicted.item()
		table[c][p] += 1
		if p == c:
			pCorrect += 1 / len(data)
	
	for i, c in enumerate(table):
		print(("%s" % classToLabel[i]).rjust(15), end="|")
		for p in c:
			print(("%d" % p).rjust(4), end="")
		print("")
	print("    %% : %0.2f" % pCorrect)

testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])




# Turn of random single kernel in first layer
indices = list(np.random.choice(NCHANNELS, 4, replace=False))
for i in indices:
	print("Disabling kernel %d in layer 1" % i)
	model.convLayers[0][0].weight.data[i].numpy().fill(0)
	model.convLayers[0][0].bias.data[i].numpy().fill(0)

indices = list(np.random.choice(NCHANNELS*NCHANNELS, 32, replace=False))
for i in indices:
	iFeature = i // NCHANNELS
	iKernel = i % NCHANNELS
	print("Disabling kernel %d of feature %d in layer 2" % (iKernel, iFeature))
	model.convLayers[1][0].weight[iFeature][iKernel].data.numpy().fill(0)
	# model.convLayers[1][0].bias[iFeature].data.numpy().fill(0)
torch.save(model.state_dict(), MODELDIR + "/modified.model")

testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])






















