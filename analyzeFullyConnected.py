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

print("Loading all model iterations..")
models = []
for file in weightFiles:
	model, _, _, _, _, _, _, _, _, _, _, _ = dataloader.loadModelFromDir(modelDir, file)
	model.eval() # Set model to evaluation mode (disables dropout layers)
	models.append(model)

# Load other parameters into global variables
model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(modelDir, weightFiles[0])

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

### Render all weights of all iterations and place them in list
renders = []
for iModel, model in enumerate(models):
	if iModel % 1 != 0:
		continue

	weights = model.fc.weight.data.numpy()
	weights = weights.reshape((NCLASSES, NCHANNELS, 4*4))
	render = render3dBarCharts(weights, 10, 2, 200)
	renders.append(render)

RENDER_HEIGHT, RENDER_WIDTH, _ = renders[0].shape

### Create sidebar with class labels
SIDEBAR_WIDTH = 200
sidebar = np.ones((RENDER_HEIGHT, SIDEBAR_WIDTH, 3), dtype=np.uint8) * 20
for i, c in enumerate(CLASSES):
	y = int(i * RENDER_HEIGHT // 6 + RENDER_HEIGHT // 12)
	cv2.putText(sidebar, c.rjust(12), (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)

writer = cv2.VideoWriter(modelDir + "/fc_weights.mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (RENDER_WIDTH+SIDEBAR_WIDTH, RENDER_HEIGHT))

### Write all renders+sidebar to video
for iRender, render in enumerate(renders):
	img = np.concatenate((sidebar, render), axis=1)
	if iRender % 10 == 0:
		cv2.imwrite(modelDir +"/weights_"+str(iRender)+".png", img)
	writer.write(img)
writer.release()

### Write last weights as image to model directory
cv2.imwrite(modelDir +"/weights_after.png", img)