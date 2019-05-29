import sys
import os
import numpy as np
from PIL import Image, ImageOps

def prepImage(im, imsize=64):
	im = ImageOps.invert(im) # Invert the image. Background becomes black, drawing becomes white
	im = im.resize((imsize, imsize), Image.LANCZOS) # Resize from (1111, 1111) to (64, 64); 
	imData = np.array(im) / 255 # Convert image to Numpy array and normalize pixel values to [0, 1]
	imData = imData.reshape(1, imsize, imsize)
	return imData

def loadAndPrepImage(imPath, imsize=64):
	im = Image.open(imPath) # Load image
	return prepImage(im, imsize)	

def loadAndPrepAllImages(nClasses = 250, nFiles = 80, imsize=64, rootfolder="./sketches_png"):
	print("\n[dataloader.py] Loading and preparing %d classes with %d images each :" % (nClasses, nFiles), end=" ", flush=True)
	# Retrieve all classes	
	classes = []
	for classDir in os.listdir(rootfolder):
		if len(classes) == nClasses:
			break
		if not os.path.isfile(rootfolder + "/" + classDir):
			classes.append(classDir)

	classes.sort() # Assure that classes will always have the same class label

	data = []
	iClasses = []
	classToLabel = ["?"] * len(classes)
	
	# Load all files for all classes
	for iC, c in enumerate(classes):
		print("%s(%d)" % (c, iC), end=", ", flush=True)
		

		classToLabel[iC] = c # Store int => class
		classDir = rootfolder + "/" + c
		# For each file in class directory
		files = os.listdir(classDir)
		for iF, f in enumerate(files[0:nFiles]): 
			file = loadAndPrepImage(classDir + "/" + f, imsize)
			data.append(file)
			iClasses.append(iC)

	print("\n[dataloader.py] Images loaded\n")
	return data, iClasses, classToLabel