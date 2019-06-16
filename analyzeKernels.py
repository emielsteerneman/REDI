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




# Numpy pretty print
np.set_printoptions(precision=2)

modelDir = dataloader.getAllModelDirs()[0]				# Get latest model from directory
weightFiles = dataloader.getAllModelWeights(modelDir)	# Get all training weights of model
weightFiles.reverse()
# weightFiles = weightFiles[-2:]

print("Loading all model iterations..")
models = []
for file in weightFiles:
	model, _, _, _, _, _, _, _, _ = dataloader.loadModelFromDir(modelDir, file)
	model.eval() # Set model to evaluation mode (disables dropout layers)
	models.append(model)

# Load other parameters into global variables
model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES = dataloader.loadModelFromDir(modelDir, weightFiles[0])

summary(model, (1, IMAGE_SIZE, IMAGE_SIZE), 1, device="cpu")




COLOURS = [(255, 127, 0), (127, 0, 255), (0, 255, 127), (127, 255, 0), (255, 0, 127), (0, 127, 255)]

def renderBarChart(values, BAR_WIDTH=2, PADDING=2, COLOUR=(255, 255, 255), FACTOR=100):
	if len(values.shape) != 1:
		print("BRUH!")

	NVALUES = values.shape[0]
	GRAPH_WIDTH = NVALUES * (BAR_WIDTH+PADDING) - PADDING
	GRAPH_HEIGHT = 150
	chart = np.ones((GRAPH_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8) * 20
	x = 0
	for v in values:
		y = int(v*FACTOR)
		cv2.rectangle(chart, (x, GRAPH_HEIGHT//2), (x + BAR_WIDTH, GRAPH_HEIGHT//2-y), COLOUR, -1)
		x += BAR_WIDTH + PADDING
	return chart

def render2dBarCharts(values, BAR_WIDTH=2, PADDING=2, FACTOR=100):
	if len(values.shape) != 2:
		print("BRUH!**2")
	## Render all bar charts
	renders = []
	for iValue, value in enumerate(values):
		render = renderBarChart(value, BAR_WIDTH, PADDING, COLOURS[iValue%len(COLOURS)], FACTOR)
		renders.append(render)
	## Calculate some values about width and height etc
	RENDER_HEIGHT, RENDER_WIDTH, _ = renders[0].shape
	GRAPH_WIDTH = values.shape[0] * (RENDER_WIDTH+BAR_WIDTH) - BAR_WIDTH
	## Create image
	chart = np.ones((RENDER_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8) * 20
	## Fill image with all the renders
	for iRender, render in enumerate(renders):
		x = iRender * (RENDER_WIDTH+BAR_WIDTH)
		chart[0:RENDER_HEIGHT, x:x+RENDER_WIDTH, :] = render
	
	return chart

def render3dBarCharts(values, BAR_WIDTH=2, PADDING=2, FACTOR=100):
	if len(values.shape) != 3:
		print("BRUH!**3")
	## Render all bar charts
	renders = []
	for iValue, value in enumerate(values):
		render = render2dBarCharts(value, BAR_WIDTH, PADDING, FACTOR)
		renders.append(render)
	## Calculate some values about width and height etc
	RENDER_HEIGHT, RENDER_WIDTH, _ = renders[0].shape
	GRAPH_HEIGHT = values.shape[0] * (RENDER_HEIGHT+BAR_WIDTH) - BAR_WIDTH
	## Create image
	chart = np.ones((GRAPH_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8) * 20
	## Fill image with all the renders
	for iRender, render in enumerate(renders):
		y = iRender * (RENDER_HEIGHT+BAR_WIDTH)
		chart[y:y+RENDER_HEIGHT, 0:RENDER_WIDTH, :] = render

	return chart

def drawKernel(weights):
	if len(weights.shape) == 3 and weights.shape[0] == 1:
		weights = weights.squeeze(axis=0)

	if len(weights.shape) != 2:
		print("Incorrect shape", weights.shape)
		return np.zeros((3, 3))
	img = cv2.resize(weights, (90, 90), interpolation=cv2.INTER_NEAREST)
	return img


MIN = 0
MAX = 0
for iModel, model in enumerate(models):
	conv1Kernels = model.convLayers[0][0].weight.data.numpy() 
	conv2Kernels = model.convLayers[1][0].weight.data.numpy()
	MIN = min(MIN, np.min(conv1Kernels))
	MIN = min(MIN, np.min(conv2Kernels))
	MAX = max(MAX, np.max(conv1Kernels))
	MAX = max(MAX, np.max(conv2Kernels))

### Render all weights of all iterations and place them in list
renders = []

for iModel, model in enumerate(models):
	conv1Kernels = model.convLayers[0][0].weight.data.numpy() 
	conv2Kernels = model.convLayers[1][0].weight.data.numpy()

	img = np.ones((900, 1030), dtype=np.uint8) * 20
	for iKernel, kernel in enumerate(conv1Kernels):
		k = drawKernel(kernel) * 255
		x, y = 20, 20 + iKernel*110
		img[y:y+90, x:x+90] = k

	for iKernels, kernels in enumerate(conv2Kernels):
		for iKernel, kernel in enumerate(kernels):
			k = drawKernel(kernel)
			k = 255 * (k - MIN) / (MAX - MIN)

			x, y = 150 + iKernel*110, 20 + iKernels*110
			img[y:y+90, x:x+90] = k
		cv2.imshow("kernels", img)
		cv2.waitKey(10)

	renders.append(img)

RENDER_HEIGHT, RENDER_WIDTH = renders[0].shape

### Create sidebar with class labels
writer = cv2.VideoWriter(modelDir + "/kernels.mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (RENDER_WIDTH, RENDER_HEIGHT))

### Write all renders+sidebar to video
for iRender, render in enumerate(renders):
	img = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)
	img[:, :, 0] = render
	img[:, :, 1] = render
	img[:, :, 2] = render
	writer.write(img)
writer.release()

### Write last weights as image to model directory
cv2.imwrite(modelDir +"/kernels_before.png", renders[0])
cv2.imwrite(modelDir +"/kernels_after.png", renders[-1])