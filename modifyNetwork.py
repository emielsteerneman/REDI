import sys
import numpy as np
import dataloader
from torchsummary import summary
import cv2
import torch
from render import renderKernels

# Numpy pretty print
np.set_printoptions(precision=2)

### Load model
# modelDir = None
# if modelDir == None:
# 	if len(sys.argv) == 1:
# 		model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
# 	else:
# 		model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(sys.argv[1])
# else:
# 	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(modelDir)

model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()

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
	return pCorrect

print("\nOG")
testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])


#### RANDOM ####
print("\nRANDOM")
FRACTION = 4
percentages = []
maxP = 0
minP = 1
for iEpoch in range(1000):
	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
	model.eval()
	indices = list(np.random.choice(NCHANNELS, NCHANNELS//FRACTION, replace=False))
	for i in indices:
		# print("Disabling kernel %d in layer 1" % i)
		model.convLayers[0][0].weight.data[i].numpy().fill(0)
		model.convLayers[0][0].bias.data[i].numpy().fill(0)

	indices = list(np.random.choice(NCHANNELS*NCHANNELS, (NCHANNELS*NCHANNELS) // FRACTION, replace=False))
	for i in indices:
		iFeature = i // NCHANNELS
		iKernel = i % NCHANNELS
		# print("Disabling kernel %d of feature %d in layer 2" % (iKernel, iFeature))
		model.convLayers[1][0].weight[iFeature][iKernel].data.numpy().fill(0)
		# model.convLayers[1][0].bias[iFeature].data.numpy().fill(0)
	p = testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])
	if maxP < p:
		img = renderKernels(model)
		cv2.imwrite(MODELDIR + "/modifiedMax.png", img)
		maxP = p
	if p < minP:
		img = renderKernels(model)
		cv2.imwrite(MODELDIR + "/modifiedMin.png", img)
		minP = p

	percentages.append(p)
	print(iEpoch, "%0.2f" % minP, "%0.2f" % maxP)
################

percentages = np.array(percentages)
print("  Min: %0.2f" % np.min(percentages))
print("  Max: %0.2f" % np.max(percentages))
print(" Mean: %0.2f" % np.mean(percentages))
print("  Var: %0.2f" % np.var(percentages))

def disable(model, feature, kernel=None):
	if kernel == None:
		model.convLayers[0][0].weight.data[feature].numpy().fill(0)
	elif kernel == "all":
		model.convLayers[1][0].weight[feature].data.numpy().fill(0)
	else:
		model.convLayers[1][0].weight[feature][kernel].data.numpy().fill(0)

#### SPECIFIC ####
model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
model.eval()

disable(model, 0)
disable(model, 2)

disable(model, 0, 0)
disable(model, 0, 1)
disable(model, 1, 1)
disable(model, 1, 6)
disable(model, 2, 2)
disable(model, 2, 3)
disable(model, 2 ,5)
disable(model, 3, 0)

disable(model, 3, 1)
disable(model, 4, 2)
disable(model, 4, 7)
disable(model, 5, 2)
disable(model, 5, 3)
disable(model, 6, 0)
disable(model, 6, 7)
disable(model, 7, 2)



print("\nCustom performance")
testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])
##################


torch.save(model.state_dict(), MODELDIR + "/modified.model")

img = renderKernels(model)
cv2.imwrite(MODELDIR + "/modified.png", img)
cv2.imshow("Kernels of model", img)
cv2.waitKey(0)





















