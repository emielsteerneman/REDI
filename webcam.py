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
# Numpy pretty print
np.set_printoptions(precision=2)

### Load model
modelDir = None
if modelDir == None:
	if len(sys.argv) == 1:
		model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES = dataloader.loadLatestModel()
	else:
		model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES = dataloader.loadModelFromDir(sys.argv[1])
else:
	model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES = dataloader.loadModelFromDir(modelDir)

# Set model to evaluation mode (disables dropout layers)
model.eval()
summary(model, (1, IMAGE_SIZE, IMAGE_SIZE), 1)

### Load class labels
predictions = {}
for c in CLASSES:
	predictions[c] = 0.0

print("Opening webcam...")
cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer = cv2.VideoWriter("/home/emiel/Desktop/REDI_.mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (WIDTH, HEIGHT))

BINARY_THRESHOLD = 130
NP_THRESHOLD = 0.3



##########################
### ADD HOOKS TO MODEL ###
convActivations  = np.zeros((len(model.convLayers), NCHANNELS))
convActivationsT = np.zeros((len(model.convLayers), NCHANNELS))
fullActivations  = np.zeros((1, NCHANNELS * 4 * 4))
fullActivationsT = np.zeros((1, NCHANNELS * 4 * 4))
# fullActivations  = np.zeros(NCHANNELS * 4 * 4)
# fullActivationsT = np.zeros(NCHANNELS * 4 * 4)

def hook_fn(iLayer, module, input, output):
	# Get data
	activations = output.data[0].numpy().copy()
	# Get average of activations
	activations = activations.max(axis=(1, 2))
	convActivations[iLayer] = activations.copy()
	convActivationsT[iLayer] += 0.1 * (convActivations[iLayer] - convActivationsT[iLayer])

def dropout_hook_fn(module, input, output):
	fullActivations = input[0].data[0].numpy()
	fullActivationsT[0] += 0.1 * (fullActivations - fullActivationsT[0])

print("Adding hooks to model..")
for iLayer, layer in enumerate(model.convLayers):
	# Ugly hack to create a closure for the variable iLayer. https://stackoverflow.com/a/2295368/2165613
	layer.register_forward_hook(	  (lambda iL : lambda module, input, output : hook_fn(iL, module, input, output))(iLayer)		)
model.drop_out.register_forward_hook(dropout_hook_fn)
### HOOKS ADDED TO MODEL ### 
############################

###################################
### EVERYTHING BARCHART RELATED ###
COLOURS = [(255, 127, 0), (127, 0, 255), (0, 255, 127), (127, 255, 0), (255, 0, 127), (0, 127, 255)]
def getColour(i):
	return COLOURS[ i % len(COLOURS) ]


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



# while True:
# 	output, _ = classifyImage(model, airplane)
# 	for p, l in enumerate(output):
# 		print(l, p)
# 	print()
# 	time.sleep(1)
# exit()



while(True):
	# time.sleep(0.5)
	# Capture frame-by-frame
	ret, frame = cap.read()

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

	### Be square or be gone
	if len(squaresFound) == 0:
		cv2.imshow("frame",frame)
		# writer.write(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		continue

	# Just grab the last square for now
	x1, y1, x2, y2, x, y, h, w = squaresFound[-1]
	cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 128, 0),2)		
	drawing = gray[y1:y2, x1:x2]

	# Classify image
	output, im = classifyImage(model, drawing)

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

	###########################
	### Visualize bar chart ###
	BAR_WIDTH, PADDING = 20, 2
	LAYER_WIDTH = NCHANNELS * (BAR_WIDTH+PADDING) - PADDING + BAR_WIDTH
	GRAPH_WIDTH = BAR_WIDTH + NLAYERS * LAYER_WIDTH
	
	chart = np.ones((400, GRAPH_WIDTH, 3), dtype=np.uint8) * 20
	for iLayer, activations in enumerate(convActivationsT):
		x = BAR_WIDTH + iLayer * LAYER_WIDTH
		for iA, a in enumerate(activations):
			y = int(a*10)
			cv2.rectangle(chart, (x, 400), (x + BAR_WIDTH, 400-y), getColour(iLayer), -1)
			x += BAR_WIDTH + PADDING
	cv2.imshow("bar chart", chart)
	###########################

	###########################
	### Visualize bar chart ###
	BAR_WIDTH, PADDING = 10, 2
	LAYER_WIDTH = 16 * (BAR_WIDTH+PADDING) - PADDING + BAR_WIDTH
	GRAPH_WIDTH = BAR_WIDTH + NCHANNELS * LAYER_WIDTH
	
	chart = np.ones((800, GRAPH_WIDTH, 3), dtype=np.uint8) * 20
	for iChannel in range(0, 8):
		pixels = fullActivationsT[0][iChannel*16 : (iChannel+1) * 16]
		x = BAR_WIDTH + iChannel * LAYER_WIDTH
		for iPixel, pixel in enumerate(pixels):
			y = int(pixel * 100)
			cv2.rectangle(chart, (x, 400), (x + BAR_WIDTH, 400-y), getColour(iChannel), -1)
			x += BAR_WIDTH + PADDING
	cv2.imshow("bar chart 2", chart)
			# print("\t", iPixel, pixel)
		# 	x = BAR_WIDTH + iChannel * LAYER_WIDTH + iPixel
		# 	for iA, a in enumerate(activations):
		# 		y = int(a*10)
		# 		cv2.rectangle(chart, (x, 400), (x + BAR_WIDTH, 400-y), getColour(iLayer), -1)
		# 		x += BAR_WIDTH + PADDING
		# cv2.imshow("bar chart", chart)
	###########################


	cv2.imshow("frame",frame)

	# writer.write(frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




