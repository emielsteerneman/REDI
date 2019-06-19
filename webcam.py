import cv2
import os
import torch
import dataloader
import pickle
from network import ConvNet
from PIL import Image
import numpy as np
import time
import sys
from torchsummary import summary
import operator
import argparse
import render

# Numpy pretty print
np.set_printoptions(precision=2)

### Load model

parser = argparse.ArgumentParser()
parser.add_argument('-d', metavar='DIRECTORY', required=False, help='Directory to load the model from')
parser.add_argument('-m', metavar='MODEL_NAME', required=False, help='Name of model to be loaded')
parser.add_argument('-w', metavar='MODEL_NAME', required=False, type=int, default=0, help='Name of model to be loaded')
args = parser.parse_args()

modelDir = args.d
modelName = args.m
print(modelDir,modelName)
if modelDir == None and modelName == None:
	print("loadLatestModel()")
	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadLatestModel()
else:
	modelDir = modelDir if modelDir is not None else dataloader.getAllModelDirs("./")[0]
	print("loadModelFromDir(" + str(modelDir) + ", " + str(modelName) + ")")
	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, MODELDIR, INDICES_TRAIN, INDICES_TEST = dataloader.loadModelFromDir(modelDir, modelName)


# Set model to evaluation mode (disables dropout layers)
model.eval()
summary(model, (1, IMAGE_SIZE, IMAGE_SIZE), 1, "cpu")

### Load class labels
predictions = {}
for c in CLASSES:
	predictions[c] = 0.0

print("Opening webcam...")
cap = cv2.VideoCapture(args.w)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer = cv2.VideoWriter("/home/emiel/Desktop/REDI_.mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (1800, 800))

BINARY_THRESHOLD = 100
NP_THRESHOLD = 0.3








from scipy import signal

def gkern(kernlen=21, std1=3, std2=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
    gkern1d2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d1, gkern1d2)
    return gkern2d

def generateGauss(xOffset=0, yOffset=0, std1=3, std2=3):
	W, H = 640, 480
	SIZE = 100
	kern = gkern(SIZE, std1, std2)
	gauss = np.ones((H, W, 3), dtype=np.uint8) * 255
	x1, x2 = W//2 - SIZE//2 + xOffset, W//2 + SIZE//2 + xOffset
	y1, y2 = H//2 - SIZE//2 + yOffset, H//2 + SIZE//2 + yOffset
	gauss[y1:y2, x1:x2, 0] = 255 - kern * 255
	gauss[y1:y2, x1:x2, 1] = 255 - kern * 255
	gauss[y1:y2, x1:x2, 2] = 255 - kern * 255
	cv2.rectangle(gauss, (120, 40), (520, 440), (0, 0, 0), 5)
	return gauss

def generateChecker(step):
	W, H = 640, 480
	checker = np.ones((H, W, 3), dtype=np.uint8) * 255
	dx, dy = step % 4, (step%16) // 4
	x, y = 157 + 82 * dx, 77 + 82 * dy
	
	cv2.rectangle(checker, (120, 40), (520, 440), (0, 0, 0), 5)
	cv2.rectangle(checker, (x+10, y+10), (x+75, y+75), (0, 0, 0), -1)
	return checker	




##########################
### ADD HOOKS TO MODEL ###
convRenders = [None] * (NLAYERS+1)
fullActivations  = np.zeros((1, NCHANNELS * 4 * 4))
fullActivationsT = np.zeros((1, NCHANNELS * 4 * 4))

def hook_fn(iLayer, module, input, output):
	activations = output.data[0].numpy().copy()
	img = render.renderConvLayer(activations, PADDING=10)
	convRenders[iLayer] = img
	# cv2.imshow("layer %d" % iLayer, img)

def fc_hook_fn(module, input, output):
	fullActivations = input[0].data[0].numpy()
	fullActivationsT[0] += 0.1 * (fullActivations - fullActivationsT[0])

print("Adding hooks to model..")
for iLayer, layer in enumerate(model.convLayers):
	# Ugly hack to create a closure for the variable iLayer. https://stackoverflow.com/a/2295368/2165613
	layer.register_forward_hook(	  (lambda iL : lambda module, input, output : hook_fn(iL, module, input, output))(iLayer)		)
model.adaptive.register_forward_hook(lambda module, input, output : hook_fn(NLAYERS, module, input, output))

model.fc.register_forward_hook(fc_hook_fn)
### HOOKS ADDED TO MODEL ### 
############################

print("Loading single image..")
airplane = dataloader.loadAndPrepImage("sketches_png/apple/322.png", imsize=IMAGE_SIZE).reshape(IMAGE_SIZE, IMAGE_SIZE)


def classifyImage(model, image):
	image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
	
	## Convert image to range [0, 1] and normalize
	im = np.array(image) / 255	# Convert image to range [0, 1]
	im -= np.min(im)			# Normalize image
	if np.max(im) != 0:			# Normalize image
		im /= np.max(im)		# Normalize image
	im = 1 - im 				# Invert image. Black drawing becomes white

	## Convert image to binary
	indices = im < NP_THRESHOLD # Threshold image
	im[indices] = 0				# Float to binary
	im[~indices]= 1				# Float to binary

	# cv2.imshow("thresh", im)

	## Convert to FloatTensor and put through network
	_input = torch.FloatTensor( im.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE) )
	output = model(_input)
	
	output = output.data[0].numpy()
	output[output < 0] = 0
	if max(output) != 0:
		output /= max(output)
	output = list(zip(output, CLASSES))

	return output, im

iFrame = 0

while(True):
	iFrame += 1
	ret, frame = cap.read()
	
	# xOffset, yOffset, std1, std2 = int(np.sin(iFrame/37) * 150), int(np.cos(iFrame/13) * 150), int(np.sin(iFrame/60)*40), int(np.cos(iFrame/93)*40)
	# frame = generateGauss(xOffset, yOffset, std1, std2)
	# frame = generateChecker(iFrame // 30)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

	### Find all squares in the image
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	squaresFound = []
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if w*h < (HEIGHT/4) * (WIDTH/4):
			continue
		if HEIGHT*0.8*WIDTH*0.8 < w*h:
			continue
		if w/h < 0.8 or h/w < 0.8:
			continue

		x1, x2 = x + w//10, x + w - w//10
		y1, y2 = y + h//10, y + h - h//10
		squaresFound.append([x1, y1, x2, y2, x, y, h, w])
		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	# cv2.imshow("binary", binary)

	### Be square or be gone. Stop if there are no squares found
	if len(squaresFound) == 0:
		cv2.imshow("frame", frame)
		cv2.imshow("binary", binary)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		continue
	### If a square is found, try closing the window "frame" and "binary" that might have been opened above
	try:
		cv2.destroyWindow("binary")
		cv2.destroyWindow("frame")
	except:
		pass

	# Just grab the last square for now
	x1, y1, x2, y2, x, y, h, w = squaresFound[-1]
	cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 128, 0), 1)		

	xStep, yStep = (x2-x1)//4, (y2-y1)//4
	for dx in range(4):
		for dy in range(4):
			dx1, dy1 = x1 + dx * xStep, y1 + dy * yStep
			cv2.rectangle(frame, (dx1, dy1), (dx1+xStep, dy1+yStep), (0, 0, 0), 1)

	drawing = gray[y1:y2, x1:x2]

	# Classify image
	output, im = classifyImage(model, drawing)

	### Draw classification
	for i, (p, l) in enumerate(output):
		predictions[l] -= (predictions[l] - p) * 0.1	
		pX, pY = 40, 40+20*i

		# Draw gray box
		rectSize = 100
		cv2.rectangle(frame, (pX, pY-12), (pX+rectSize, pY+5), (200, 200, 200), -1)
		# Draw green box
		rectSize = max(0, int(predictions[l] * 100))
		cv2.rectangle(frame, (pX, pY-12), (pX+rectSize, pY+5), (0, 255, 0), -1)
		# Draw blue line
		rectSize = int(p * 100)
		cv2.rectangle(frame, (pX, pY+4), (pX+rectSize, pY+5), (255, 0, 0), -1)
		# Write label
		cv2.putText(frame, l, (pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), lineType=cv2.LINE_AA)

	# Add preprocessed image to frame
	im *= 255
	x0, y0 = 40, HEIGHT - IMAGE_SIZE - 40
	x1, y1 = x0 + IMAGE_SIZE, y0 + IMAGE_SIZE
	frame[y0:y1, x0:x1, 0] = im
	frame[y0:y1, x0:x1, 1] = im
	frame[y0:y1, x0:x1, 2] = im



	########################################
	# vvvvvvvvvv VISUALISATIONS vvvvvvvvvv #
	########################################

	###########################################################################
	### Visualize output of last convolutional layer / adaptive layer / input of fully connected layer ###
	pixels = fullActivationsT[0].reshape((NCHANNELS, 4*4))
	barChartFc = render.render2dBarCharts(pixels, BAR_WIDTH=3, PADDING=2, FACTOR=20, GRAPH_HEIGHT=200)
	barChartFc = np.rot90(barChartFc, 3)
	# cv2.imshow("Input of fully connected layer", barChartFc)
	###########################################################################

	activeClass = max(predictions.items(), key=operator.itemgetter(1))[0]
	iActiveClass = CLASSES.index(activeClass)

	#########################################
	### Visualize weights of active class ###
	classWeights = model.fc.weight[iActiveClass].data.numpy().reshape((NCHANNELS, 4*4))
	barChartWeights = render.render2dBarCharts(classWeights, BAR_WIDTH=3, PADDING=2, FACTOR=100, GRAPH_HEIGHT=100, COLOUR=(255, 255, 255))
	barChartWeights = np.rot90(barChartWeights, 3)
	# cv2.imshow("Weights of active class", barchartWeights)
	#########################################

	#####################################################
	### Visualize product of inputs and class weights ###
	classWeights = model.fc.weight[iActiveClass].data.numpy()
	inputsAndWeights = (fullActivationsT[0] * classWeights).reshape((NCHANNELS, 4*4))
	barChartFcWeights = render.render2dBarCharts(inputsAndWeights, BAR_WIDTH=3, PADDING=2, FACTOR=200, GRAPH_HEIGHT=200)
	barChartFcWeights = np.rot90(barChartFcWeights, 3)
	# cv2.imshow("Inputs times weights", barChartFcWeights)
	#####################################################











	FINAL_WIDTH = 1800
	FINAL_HEIGHT = 800
	final = np.ones((FINAL_HEIGHT, FINAL_WIDTH, 3), dtype=np.uint8) * 20

	## Add convolutional renders
	for iConvRender, convRender in enumerate(convRenders):
		RENDER_HEIGHT, RENDER_WIDTH, _ = convRender.shape
		x = 10 + iConvRender * (RENDER_WIDTH + 50)
		y = FINAL_HEIGHT//2 - RENDER_HEIGHT//2
		final[y:y+RENDER_HEIGHT, x:x+RENDER_WIDTH, :] = convRender
		txt = "Layer %d" % iConvRender
		cv2.putText(final, txt, (x+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

	## Add input of fully connected layer
	x = 400
	y = FINAL_HEIGHT//2 - barChartFc.shape[0] // 2
	final[y:y+barChartFc.shape[0], x:x+barChartFc.shape[1], :] = barChartFc
	txt = "Fully Connected"
	cv2.putText(final, txt, (x+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

	x = 600
	y = FINAL_HEIGHT//2 - barChartFcWeights.shape[0] // 2
	final[y:y+barChartWeights.shape[0], x:x+barChartWeights.shape[1], :] = barChartWeights
	txt = "Weights '%s'" % activeClass
	cv2.putText(final, txt, (x+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

	x = 700
	y = FINAL_HEIGHT//2 - barChartFcWeights.shape[0] // 2
	final[y:y+barChartFcWeights.shape[0], x:x+barChartFcWeights.shape[1], :] = barChartFcWeights
	txt = "FC * Weights"
	cv2.putText(final, txt, (x+80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)	

	final[-HEIGHT:, -WIDTH:, :] = frame

	cv2.imshow("final", final)
	#########################################



	# cv2.imshow("frame",frame)
	# writer.write(final)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
# writer.release()
cap.release()
cv2.destroyAllWindows()




