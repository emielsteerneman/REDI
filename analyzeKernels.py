import cv2
import os
# import torch
import dataloader
# import pickle
# from network import ConvNet
# from PIL import Image
import numpy as np
# import time
# import sys
from torchsummary import summary

from render import renderKernels


# Numpy pretty print
np.set_printoptions(precision=2)

modelDir = dataloader.getAllModelDirs()[0]				# Get latest model from directory
weightFiles = dataloader.getAllModelWeights(modelDir)	# Get all training weights of model
weightFiles.reverse()

print("Loading all model iterations..")
models = []
for file in weightFiles:
	model, _, _, _, _, _, _, _, _, _, _, _  = dataloader.loadModelFromDir(modelDir, file)
	model.eval() # Set model to evaluation mode (disables dropout layers)
	models.append(model)

# Load other parameters into global variables
model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(modelDir, weightFiles[0])

summary(model, (1, IMAGE_SIZE, IMAGE_SIZE), 1, device="cpu")

COLOURS = [(255, 127, 0), (127, 0, 255), (0, 255, 127), (127, 255, 0), (255, 0, 127), (0, 127, 255)]

MIN = 0
MAX = 0
for iModel, model in enumerate(models):
	conv1Kernels = model.convLayers[0][0].weight.data.numpy() 
	conv2Kernels = model.convLayers[1][0].weight.data.numpy()
	MIN = min(MIN, np.min(conv1Kernels), np.min(conv2Kernels))
	MAX = max(MAX, np.max(conv1Kernels), np.max(conv2Kernels))

### Render all weights of all iterations and place them in list
renders = []

for iModel, model in enumerate(models):
	renders.append(renderKernels(model, MIN, MAX))

RENDER_HEIGHT, RENDER_WIDTH = renders[0].shape

### Create sidebar with class labels
writer = cv2.VideoWriter(modelDir + "/kernels.mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (RENDER_WIDTH, RENDER_HEIGHT))

### Colour of 0
c = int(-MIN * 256 / (MAX - MIN))

### Write all renders+sidebar to video
for iRender, render in enumerate(renders):
	img = np.ones((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8) * 20
	
	# Add bar
	# bw = 500
	# bx = (RENDER_WIDTH - bw) // 2
	# by = 15
	# cv2.rectangle(img, (bx, by), (bx+bw, by+20), (c, c, c), -1)
	# cv2.putText(img, "%0.3f" % MIN, (bx-100, by+18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)
	# cv2.putText(img, "%0.3f" % MAX, (bx+bw+5, by+18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)

	img[:, :, 0] = render
	img[:, :, 1] = render
	img[:, :, 2] = render

	writer.write(img)
	
	if iRender == 0:
		cv2.imwrite(modelDir +"/kernels_before.png", img)

writer.release()

### Write last weights as image to model directory
cv2.imwrite(modelDir +"/kernels_after.png", img)
