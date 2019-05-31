import cv2
import os
import torch
import dataloader
import pickle
from network import ConvNet
from PIL import Image
import numpy as np
import time

modelDir = "./model_190531-114303_128_10_80"
model, date, IMAGE_SIZE, NCLASSES, NFILES = dataloader.loadModelFromDir(modelDir)

# with open("classToLabel.pickle", "rb") as pickle_file:
#    classToLabel = pickle.load(pickle_file)
_, _, classToLabel = dataloader.loadAndPrepAllImages(NCLASSES, 1, 1)
classToLabel = classToLabel[0:NCLASSES]
predictions = {}
for c in classToLabel:
	predictions[c] = 0.0

cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer = cv2.VideoWriter("/home/emiel/Desktop/REDI.mp4", cv2.VideoWriter_fourcc(*'XVID'), 20, (WIDTH, HEIGHT))

def classifyImage(model, image):
	image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
	im = np.array(image) / 255
	im -= np.min(im)
	im /= np.max(im)
	im = 1 - im

	indices = im < 0.2
	im[indices] = 0
	im[~indices]= 1

	_input = torch.FloatTensor( im.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE) )
	output = model(_input)
	
	output = list(output.data[0].numpy())
	output = list(zip(output, classToLabel))
	# output.sort(reverse=True)

	return output, im


while(True):
	time.sleep(0.1)
	# Capture frame-by-frame
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if w*h < (HEIGHT/4) * (HEIGHT/4):
			continue
		if w/h < 0.8 or h/w < 0.8:
			continue

		x1, x2 = x + w//10, x + w - w//10
		y1, y2 = y + h//10, y + h - h//10
		
		drawing = gray[y1:y2, x1:x2]
		output, im = classifyImage(model, drawing)
		minP = -5#min([p for p, l in output])

		for i, (p, l) in enumerate(output):
			predictions[l] -= (predictions[l] - p) * 0.1

			rectSize = int(predictions[l] - minP) * 5
			pX, pY = 40, 40+20*i

			cv2.rectangle(frame, (pX, pY-12), (pX+rectSize, pY+6), (0, 255, 0), -1)
			cv2.putText(frame, l, (pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), lineType=cv2.LINE_AA)

		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		im *= 255
		x0, y0 = 40, HEIGHT - IMAGE_SIZE - 40
		x1, y1 = x0 + IMAGE_SIZE, y0 + IMAGE_SIZE

		frame[y0:y1, x0:x1, 0] = im
		frame[y0:y1, x0:x1, 1] = im
		frame[y0:y1, x0:x1, 2] = im

	cv2.imshow("frame",frame)
	# writer.write(frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




