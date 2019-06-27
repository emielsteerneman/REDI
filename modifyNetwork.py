import sys
import numpy as np
import dataloader
from torchsummary import summary
import cv2
import torch
from render import renderKernels, render3dBarCharts

# Numpy pretty print
np.set_printoptions(precision=4)

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
todisable = ["fc"]
maxP = 0
minP = 1
for iEpoch in range(1000):
	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
	model.eval()
	if "kernels" in todisable:
		indices = list(np.random.choice(NCHANNELS, 1, replace=False))
		for i in indices:
			# print("Disabling kernel %d in layer 1" % i)
			model.convLayers[0][0].weight.data[i].numpy().fill(0)
			# model.convLayers[0][0].bias.data[i].numpy().fill(0)

		indices = list(np.random.choice(NCHANNELS*NCHANNELS, (NCHANNELS*NCHANNELS) // FRACTION, replace=False))
		for i in indices:
			iFeature = i // NCHANNELS
			iKernel = i % NCHANNELS
			# print("Disabling kernel %d of feature %d in layer 2" % (iKernel, iFeature))
			model.convLayers[1][0].weight[iFeature][iKernel].data.numpy().fill(0)
			# model.convLayers[1][0].bias[iFeature].data.numpy().fill(0)
	
	if "fc" in todisable:
		indices = list(np.random.choice(128*NCLASSES, (128*NCLASSES) // (4*FRACTION), replace=False))
		for i in indices:
			c = i//128
			w = i%128
			model.fc.weight.data.numpy()[c][w] *= -1


	p = testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])
	
	if maxP < p:
		if "kernels" in todisable:
			img = renderKernels(model)
			cv2.imwrite(MODELDIR + "/modifiedMax.png", img)
		if "fc" in todisable:
			weights = model.fc.weight.data.numpy()
			weights = weights.reshape((NCLASSES, NCHANNELS, 4*4))
			img = render3dBarCharts(weights, 10, 2, 200)
			cv2.imwrite(MODELDIR + "/weightsMax.png", img)
		maxP = p
	if p < minP:
		if "kernels" in todisable:
			img = renderKernels(model)
			cv2.imwrite(MODELDIR + "/modifiedMin.png", img)
		if "fc" in todisable:
			weights = model.fc.weight.data.numpy()
			weights = weights.reshape((NCLASSES, NCHANNELS, 4*4))
			img = render3dBarCharts(weights, 10, 2, 200)
			cv2.imwrite(MODELDIR + "/weightsMin.png", img)
		minP = p

	percentages.append(p)
	print(iEpoch, "%0.2f" % minP, "%0.2f" % maxP)
################

percentages = np.array(percentages)
print("  Min: %0.2f" % np.min(percentages))
print("  Max: %0.2f" % np.max(percentages))
print(" Mean: %0.3f" % np.mean(percentages))
print("  Var: %0.4f" % np.var(percentages))

def disable(model, row, column=None):
	if column == None:
		model.convLayers[0][0].weight.data[row].numpy().fill(0)
	elif row == "all":
		model.convLayers[1][0].weight[column].data.numpy().fill(0)
	else:
		model.convLayers[1][0].weight[column][row].data.numpy().fill(0)

def flip_weights(model, c, kernel = None, subweight=None):
	if kernel == None and subweight == None:
		model.fc.weight.data.numpy()[c] *= -1.0
	elif subweight == None:
		model.fc.weight.data.numpy()[c][kernel*16:(kernel+1)*16] *= -1.0
	elif kernel == None:
		model.fc.weight.data.numpy()[c][subweight] *= -1.0
	else:
		model.fc.weight.data.numpy()[c][kernel*16+subweight] *= -1.0

#### SPECIFIC ####
model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
model.eval()



flip_weights(model, 0, 0, 4)
flip_weights(model, 0, 1, 6)
flip_weights(model, 0, 1, 7)
flip_weights(model, 0, 2, 9)
flip_weights(model, 0, 3, 14)
flip_weights(model, 0, 4, 6)
flip_weights(model, 0, 5, 13)
flip_weights(model, 0, 7, 1)
flip_weights(model, 1, 0, 6)
flip_weights(model, 1, 1, 14)
flip_weights(model, 1, 2, 4)
flip_weights(model, 1, 3, 1)
flip_weights(model, 1, 4, 6)
flip_weights(model, 1, 5, 1)
flip_weights(model, 1, 3, 13)
flip_weights(model, 1, 2, 10)
flip_weights(model, 2, 0, 6)
flip_weights(model, 2, 0, 11)
flip_weights(model, 2, 1, 13)
flip_weights(model, 2, 3, 2)
flip_weights(model, 2, 3, 3)
flip_weights(model, 2, 4, 5)
flip_weights(model, 2, 5, 13)
flip_weights(model, 2, 6, 14)
flip_weights(model, 3, 1, 0)
flip_weights(model, 3, 1, 14)
flip_weights(model, 3, 2, 0)
flip_weights(model, 3, 2, 7)
flip_weights(model, 3, 3, 14)
flip_weights(model, 3, 4, 5)
flip_weights(model, 3, 5, 3)
flip_weights(model, 3, 6, 0)
flip_weights(model, 4, 0, 0)
flip_weights(model, 4, 1, 14)
flip_weights(model, 4, 2, 4)
flip_weights(model, 4, 2, 15)
flip_weights(model, 4, 3, 14)
flip_weights(model, 4, 4, 14)
flip_weights(model, 4, 5, 15)
flip_weights(model, 4, 6, 4)
flip_weights(model, 5, 0, 10)
flip_weights(model, 5, 0, 12)
flip_weights(model, 5, 2, 1)
flip_weights(model, 5, 2, 6)
flip_weights(model, 5, 3, 8)
flip_weights(model, 5, 3, 1)
flip_weights(model, 5, 6, 6)
flip_weights(model, 5, 6, 7)


# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 0, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 1, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 2, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 3, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 4, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)
# flip_weights(model, 5, 0, 0)




# all = "all"

# disable(model, 1)

# disable(model, 3, 2)
# disable(model, 5, 2)
# disable(model, 0, 4)
# disable(model, 0, 3)
# disable(model, 5, 0)
# disable(model, 3, 1)
# disable(model, 3, 3)
# disable(model, 5, 7)
# disable(model, 7, 3)
# disable(model, 1, 5)
# disable(model, 1, 7)
# disable(model, 2, 2)
# disable(model, 2, 6)
# disable(model, 6, 1)
# disable(model, 6, 3)
# disable(model, 4, 5)

# disable(model, 2)
# disable(model, "all", 4)
# disable(model, 1, 7)
# disable(model, 2, 7)
# disable(model, 3, 6)

# disable(model, all, 7)
# disable_weights(model, 0, 7)
# disable_weights(model, 1, 7)
# disable_weights(model, 2, 7)
# disable_weights(model, 3, 7)
# disable_weights(model, 4, 7)
# disable_weights(model, 5, 7)



print("\nCustom performance")
testPerformance(model, data[INDICES_TEST], iClasses[INDICES_TEST])
##################


torch.save(model.state_dict(), MODELDIR + "/modified.model")

img = renderKernels(model)
cv2.imwrite(MODELDIR + "/modified.png", img)
cv2.imshow("Kernels of model", img)

weights = model.fc.weight.data.numpy()
weights = weights.reshape((NCLASSES, NCHANNELS, 4*4))
img = render3dBarCharts(weights, 10, 2, 200)
cv2.imwrite(MODELDIR + "/modifiedWeights.png", img)
cv2.imshow("Weights of model", img)

cv2.waitKey(0)





















