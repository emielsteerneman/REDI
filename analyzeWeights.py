import dataloader
import torch
import cv2
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt 

def enlarge(_im):
	im = _im.copy()
	return cv2.resize(im, (500, 500), interpolation=cv2.INTER_CUBIC)

def normalize(_im):
	im = _im.copy()
	im -= np.min(im)
	im /= np.max(im)
	return im

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
        # self.features = torch.tensor(output,requires_grad=True)
        # self.features = output.clone().detach().requires_grad_(True)
    def close(self):
        self.hook.remove()

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelDir = "model_190602-122056_128_10_80"
model, date, IMAGE_SIZE, NCLASSES, NFILES = dataloader.loadModelFromDir(modelDir)

model.to(device)

summary(model, (1, 128, 128), 1)

for name, param in model.named_parameters():
	print(name.ljust(20), param.size())

img = dataloader.loadAndPrepImage("sketches_png/radio/13474.png", IMAGE_SIZE)
imgTensor = torch.FloatTensor(img.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)).cuda()

# cv2.imshow("img", img.reshape(IMAGE_SIZE, IMAGE_SIZE)*255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########################################
### Visualize image after first filter ###
##########################################
# accumulate = np.zeros((280, 1120), dtype=np.uint8)
# output = model.layer1(imgTensor)
# images = output.cpu().data[0].numpy()
# for i, image in enumerate(images):
# 	dx, dy = i % 16, i // 16
# 	px, py = dx * 70, dy * 70
# 	accumulate[py:py+64, px:px+64] = normalize(image)*255
# cv2.imshow("accumulate", accumulate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



##########################################
### Visualize feature filters manually ###
##########################################
# lyr = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2)).to(device)
# accumulate = np.zeros((IMAGE_SIZE*8, IMAGE_SIZE*8), dtype=np.float32)
# for iFeature in range(0, 64):
# 	with torch.no_grad(): # see https://discuss.pytorch.org/t/layer-weight-vs-weight-data/24271/2?u=ptrblck
# 		lyr[0].weight[0] = model.layer1[0].weight[iFeature]
# 		lyr[0].bias[0]   = model.layer1[0].bias[iFeature]

# 	dx, dy = iFeature % 8, iFeature // 8
# 	px, py = dx*IMAGE_SIZE, dy*IMAGE_SIZE
	
# 	noise = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True, device="cuda")
# 	noise.data = noise.data / 10 + 0.45

# 	optimizer = torch.optim.Adam([noise], lr=0.01, weight_decay=1e-6)
# 	for i in range(0, 2500):
# 		out = lyr(noise)		
# 		loss = -out.mean()		
# 		optimizer.zero_grad()	
# 		loss.backward()			
# 		optimizer.step()
		
# 		### Visualize evolution of noise
# 		# if i % 10 == 0:
# 		# 	cv2.imshow("noiseNorm", enlarge(normalize(noise.cpu()[0][0].data.numpy())))
# 		# if cv2.waitKey(1) & 0xFF == ord('q'):
# 		# 	exit()

# 	### Normalize data to create a clearer image
# 	data = noise.cpu().data.numpy().reshape(IMAGE_SIZE, IMAGE_SIZE)
# 	q10, q90 = np.percentile(data, [10, 90])
# 	data[data < q10] = q10
# 	data[q90 < data] = q90
# 	data -= np.min(data)
# 	data /= np.max(data)
# 	accumulate[py:py+128, px:px+128] = data
	
# 	cv2.imshow("accumulate", accumulate)
# 	cv2.waitKey(1)

# 	### Show histogram of pixels
# 	# hist, bins = np.histogram(data, bins=np.arange(np.min(data), np.max(data), 0.01))
# 	# plt.bar(bins[:-1], hist, width=1)
# 	# plt.ylim(top=max(hist)*1.5)
# 	# plt.show()
# print("%s/accumulate_%d.png" % (modelDir, 0))
# print(np.min(accumulate), np.max(accumulate))
# cv2.imwrite("%s/accumulate_manual_%d.png" % (modelDir, 0), accumulate*255)
# cv2.waitKey(0)
##########################################
##########################################
##########################################



##########################################
### Visualize feature filter with hook ###
##########################################
for iLayer, layer in enumerate(model.convLayers):
	if iLayer != 0:
		continue
	print("Layer %d" % iLayer)
	accumulate = np.zeros((IMAGE_SIZE*8, IMAGE_SIZE*8), dtype=np.float32)
	for iFeature in range(0, 64):

		STEPS = 12
		INITIAL_SIZE = 128

		dx, dy = iFeature % 8, iFeature // 8
		px, py = dx*IMAGE_SIZE, dy*IMAGE_SIZE
		
		noise = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True, device="cuda")
		noise.data = noise.data / 10 + 0.45
		
		optimizer = torch.optim.Adam([noise], lr=0.01, weight_decay=1e-6)
		sf = SaveFeatures(model.convLayers[iLayer])
		
		for iStep in range(0, STEPS):
			size = int(INITIAL_SIZE * 1.22**iStep)
			print(iStep, "%dx%d" % (size, size))

			# noise = noise.cpu().data.numpy()[0][0]
			# noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
			# noise = noise.reshape(1, 1, size, size)
			# noise = torch.FloatTensor(noise).cuda()
			# noise.requires_grad=True
			optimizer = torch.optim.Adam([noise], lr=0.01, weight_decay=1e-6)

			for i in range(0, 50):

				# if i % IMAGE_SIZE == 0 and 0 < i:
				# 	print("Resizing", i, i // STEPS, noise.size())
				# 	SIZE = (i+IMAGE_SIZE) // STEPS
				# 	noise = noise.cpu().data.numpy()[0][0]
				# 	# print(noise)
				# 	print(noise.shape)
				# 	noise = cv2.resize(noise, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
				# 	noise = noise.reshape(1, 1, SIZE, SIZE)
				# 	print(noise.shape)
				# 	noise = torch.FloatTensor(noise).cuda()


				model(noise)
				loss = -sf.features[0][iFeature].mean()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				## Visualize evolution of noise
				if i % 1 == 0:
					cv2.imshow("noiseNorm", enlarge(normalize(noise.cpu()[0][0].data.numpy())))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					exit()
		sf.close()

		# cv2.imshow("finalNoise", enlarge(normalize(noise.cpu()[0][0].data.numpy())))
		### Normalize data to create a clearer image
		data = noise.cpu().data.numpy()[0][0]
		data = cv2.resize(data, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
		q10, q90 = np.percentile(data, [10, 90])
		data[data < q10] = q10
		data[q90 < data] = q90
		data -= np.min(data)
		data /= np.max(data)
		accumulate[py:py+128, px:px+128] = data
		
		cv2.imshow("accumulate %d" % iLayer, accumulate)
		cv2.waitKey(1)

	cv2.imwrite("%s/accumulate_%d.png" % (modelDir, iLayer), accumulate)
cv2.waitKey(0)
##########################################
##########################################
##########################################
exit()


out = lyr(imgTensor)
print(out.size())
imgOut = out.data[0][0].numpy()

cv2.imshow("Second", enlarge(normalize(imgOut)))
cv2.waitKey(0)
cv2.destroyAllWindows()


exit()

w = model.layer1[0].weight
# print(w)
# print(w.shape)



# 485 -> 485 padding left and top
# 60 per block
# 	10 padding right and bottom
# 	10 per pixel

def drawTensor(t, w, h):
	kernel = t.detach().numpy().copy()
	kernel -= np.min(kernel)
	kernel *= 255
	assert kernel.ndim == 2, "Matrix does not have 2 dimensions"
	return cv2.resize(kernel, (w, h), interpolation=cv2.INTER_NEAREST)


img = np.zeros((490, 490), dtype=np.uint8)

for i, x in enumerate(w):
	dx, dy = i % 8, i // 8

	px, py = 10 + dx * 60, 10 + dy * 60
	img[py:py+50, px:px+50] = drawTensor(x[0], 50, 50)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()













def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
    sz = self.size
    img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255  # generate random image
    activations = SaveFeatures(list(self.model.children())[layer])  # register hook

    for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
        train_tfms, val_tfms = tfms_from_model(vgg16, sz)
        img_var = V(val_tfms(img)[None], requires_grad=True)  # convert image to Variable that requires grad
        optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
        for n in range(opt_steps):  # optimize pixel values for opt_steps times
            optimizer.zero_grad()
            self.model(img_var)
            loss = -activations.features[0, filter].mean()
            loss.backward()
            optimizer.step()
        img = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1,2,0))
        self.output = img
        sz = int(self.upscaling_factor * sz)  # calculate new image size
        img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
        if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
    self.save(layer, filter)
    activations.close()














    














