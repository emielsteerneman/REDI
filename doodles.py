import os
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime

print("\n\nDoodle CNN")
print("Programmed for pytorch v1.0.0")

IMAGE_SIZE = 64

# Image.fromarray(imData * 255).resize((1000, 1000)).show()

def prepImage(im):
	im = ImageOps.invert(im) # Invert the image. Background becomes black, drawing becomes white
	im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS) # Resize from (1111, 1111) to (64, 64); 
	imData = np.array(im) / 255 # Convert image to Numpy array and normalize pixel values to [0, 1]
	imData = imData.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
	return imData

def loadAndPrepImage(imPath):
	im = Image.open(imPath) # Load image
	return prepImage(im)	

def loadAndPrepAllImages(nClasses = 250, nFiles = 80):
	print("\nLoading and preparing %d classes with %d images each.." % (nClasses, nFiles))
	# Retrieve all classes	
	classes = []
	for classDir in os.listdir("./sketches_png/")[0:nClasses]:
		if not os.path.isfile("./sketches_png/" + classDir):
			classes.append(classDir)

	classes.sort() # Assure that classes will always have the same class label

	data = []
	iClasses = []
	classToLabel = ["?"] * len(classes)
	
	# Load all files for all classes
	for iC, c in enumerate(classes):
		print(" ", iC, c)
		classToLabel[iC] = c # Store int => class
		classDir = "./sketches_png/" + c
		# For each file in class directory
		for iF, f in enumerate(os.listdir(classDir)[0:nFiles]): 
			file = loadAndPrepImage(classDir + "/" + f)
			data.append(file)
			iClasses.append(iC)

	return data, iClasses, classToLabel

### https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
class ConvNet(nn.Module):
	def __init__(self, nClasses, imageSize):
		super(ConvNet, self).__init__()
		# 1 128x128 image goes in, 32 64x64 images come out
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
			# nn.ReLU())
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		imageSize = imageSize // 2

		# 32 64x64 images go in, 64 32x32 image comes out
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
			# nn.ReLU())
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		imageSize = imageSize // 2

		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(in_features=imageSize * imageSize * 64, out_features=(imageSize//4) * (imageSize//4) * 64)
		self.fc2 = nn.Linear(in_features=(imageSize//4) * (imageSize//4) * 64, out_features=nClasses)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

#####################################################################################
#####################################################################################
#####################################################################################

NCLASSES = 3
NFILES = 80

data, iClasses, classToLabel = loadAndPrepAllImages(NCLASSES, NFILES)


NTRAIN = int(NCLASSES * NFILES * 0.8)
NTEST = NCLASSES * NFILES - NTRAIN

indicesTrain = list(np.random.choice(NCLASSES * NFILES, NTRAIN, replace=False))
indicesTest = [i for i in range(0, NCLASSES * NFILES) if i not in indicesTrain]

dataTrain = [data[i] for i in indicesTrain]
iClassesTrain = [iClasses[i] for i in indicesTrain]
labelsTrain = torch.FloatTensor(iClassesTrain).long() 

dataTest = [data[i] for i in indicesTest]
iClassesTest = [iClasses[i] for i in indicesTest]
labelsTest = torch.FloatTensor(iClassesTest).long() 
## torch.FloatTensor() fixes : AttributeError: 'list' object has no attribute 'size'
## .long() fixes : RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target'

model = ConvNet(NCLASSES, IMAGE_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining CNN..")
for i in range(0, 20):
	# Run the forward pass
	y = model(torch.FloatTensor(dataTrain))
	loss = criterion(y, labelsTrain)

	# Backprop and perform Adam optimisation
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# Track the accuracy
	_, predicted = torch.max(y.data, 1)
	correct = (predicted == labelsTrain).sum().item()
	print("  iter=%d" % i, "acc=%0.2f" % (correct / NTRAIN), "loss=%0.2f" % loss)
	if(correct == NTRAIN and loss < 0.01):
		break

print("\nTesting CNN..")
y = model(torch.FloatTensor(dataTest))
_, predicted = torch.max(y.data, 1)
correct = (predicted == labelsTest).sum().item()
print("Accuracy : %0.2f" % (correct / NTEST))

## Storing model
nowStr = datetime.now().strftime("%y%m%d-%H%M%S")
modelPath = "./model_" + nowStr + "_%dx%d_nclasses=%d_acc=%0.2f.model" % (IMAGE_SIZE, IMAGE_SIZE, NCLASSES, correct / NTEST)
torch.save(model.state_dict(), modelPath)


exit()
















