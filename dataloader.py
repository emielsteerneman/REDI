import sys
import os
import numpy as np
from PIL import Image, ImageOps
import torch
import importlib

def prepImage(im, imsize=64):
	im = ImageOps.invert(im) # Invert the image. Background becomes black, drawing becomes white
	
	im = im.resize((imsize, imsize), Image.LANCZOS) # Resize from (1111, 1111) to (64, 64); 
	imData = np.array(im) / 255 # Convert image to Numpy array and normalize pixel values to [0, 1]
	
	# Image.fromarray(imData * 255).resize((1000, 1000)).show()
	
	imData -= np.min(imData)
	imData /= np.max(imData)

	indices = imData < 0.1
	imData[indices] = 0
	imData[~indices]= 1

	# Image.fromarray(imData * 255).resize((1000, 1000)).show()
	
	imData = imData.reshape(1, imsize, imsize)
	return imData

def loadAndPrepImage(imPath, imsize=64):
	im = Image.open(imPath).convert('L') # Load image

	return prepImage(im, imsize)	

def loadAndPrepAllImages(nClasses = None, nFiles = 80, imsize=64, rootfolder="./sketches_png", classes=None):
	if nClasses == None and classes == None:
		nClasses = 250
	
	if nClasses == None and classes != None:
		nClasses = len(classes)

	if nClasses != None and classes != None:
		if nClasses != len(classes):	
			print("Error! nClasses(%d) != len(classes)(%d)" % (nClasses, len(classes)))
			exit()

	print("\n[dataloader] Loading and preparing %d classes with %d images each :" % (nClasses, nFiles), end=" ", flush=True)
	if classes == None:
		# Retrieve all classes	
		classes = []
		for classDir in os.listdir(rootfolder):
			if not os.path.isfile(rootfolder + "/" + classDir):
				classes.append(classDir)
		classes.sort()
		classes = classes[:nClasses]

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

	print("\n[dataloader] Images loaded\n")
	return data, iClasses, classToLabel

def getAllModelDirs(folder="./"):
	models = list(filter(lambda f : f.startswith("model_"), os.listdir(folder)))
	models.sort(reverse=True)
	return models

def getAllModelWeights(folder="./"):
	weights = os.listdir(folder)
	weights = list(filter(lambda x : x[0].isdigit() and x.endswith(".model"), weights))
	weights.sort(key=lambda x : int(x.split("_")[0]), reverse=True)
	return weights

def loadModelFromDir(modelDir, weightsFile=None):
	network = importlib.import_module(".network", package=modelDir)

	### Retrieve information from directory name (model_190530-011512_128_250_80)
	[_, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE] = modelDir.split("_")
	NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE = int(NCLASSES), int(NFILES), int(NBATCHES), int(NLAYERS), int(NCHANNELS), int(IMAGE_SIZE)
	### Load classes
	CLASSES = open(modelDir + "/classes.txt").read().split(" ")
	### Get latest weights
	if weightsFile is None:
		weightsFile = getAllModelWeights(modelDir)[0]
	print("[dataloader] Loading weights from " + modelDir + "/" + weightsFile)
	modelWeights = torch.load(modelDir + "/" + weightsFile)
	### Create model and restore weights
	model = network.ConvNet(NCLASSES, imageSize=IMAGE_SIZE, nConvLayers=NLAYERS, nchannels=NCHANNELS)
	model.load_state_dict(modelWeights)
	
	INDICES_TRAIN, INDICES_TEST = loadIndicesFromDir(modelDir)

	return model, date, NCLASSES, NFILES, NBATCHES, NLAYERS, NCHANNELS, IMAGE_SIZE, CLASSES, modelDir, INDICES_TRAIN, INDICES_TEST

def loadIndicesFromDir(modelDir):
	f = open(modelDir + "/indices.txt", "r")
	indicesStr = f.read()
	f.close()
	indicesStr = indicesStr.split("\n")
	indicesTrain = [int(x) for x in indicesStr[0].split(" ")[1:]]
	indicesTest  = [int(x) for x in indicesStr[1].split(" ")[1:]]
	return [indicesTrain, indicesTest]

def loadLatestModel(folder="./"):
	return loadModelFromDir(getAllModelDirs(folder)[0])
