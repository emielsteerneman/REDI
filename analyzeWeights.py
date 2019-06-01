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
        self.features = torch.tensor(output,requires_grad=True)
        # self.features = output.clone().detach().requires_grad_(True)
    def close(self):
        self.hook.remove()

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelDir = "model_190531-025542_128_10_80"
model, date, IMAGE_SIZE, NCLASSES, NFILES = dataloader.loadModelFromDir(modelDir)

model.to(device)

summary(model, (1, 128, 128), 1)
exit()
for name, param in model.named_parameters():
	print(name.ljust(20), param.size())

img = dataloader.loadAndPrepImage("sketches_png/radio/13474.png", IMAGE_SIZE)
imgTensor = torch.FloatTensor(img.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE))

# cv2.imshow("img", img.reshape(IMAGE_SIZE, IMAGE_SIZE)*255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########################################
### Visualize image after first filter ###
##########################################
# accumulate = np.zeros((280, 1120), dtype=np.uint8)
# output = model.layer1(imgTensor)
# images = output.data[0].numpy()
# for i, image in enumerate(images):
# 	dx, dy = i % 16, i // 16
# 	px, py = dx * 70, dy * 70
# 	accumulate[py:py+64, px:px+64] = normalize(image)*255
# cv2.imshow("accumulate", accumulate)
# cv2.imshow("First", enlarge(normalize(images[0])))


##########################################
### Visualize feature filters manually ###
##########################################
# lyr = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2), torch.nn.ReLU()).to(device)
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
# 	for i in range(0, 1000):
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
# 	data -= q10
# 	data /= q10-q90

# 	accumulate[py:py+128, px:px+128] = data
# 	cv2.imshow("accumulate", accumulate)
# 	cv2.waitKey(1)

# 	### Show histogram of pixels
# 	# hist, bins = np.histogram(data, bins=np.arange(np.min(data), np.max(data), 0.01))
# 	# plt.bar(bins[:-1], hist, width=1)
# 	# plt.ylim(top=max(hist)*1.5)
# 	# plt.show()
# cv2.waitKey(0)	
##########################################
##########################################
##########################################

exit()

### Visualize feature filter with hook
noise = torch.rand(1, 1, 128, 128, requires_grad=True)
noise.data = noise.data / 10 + 0.45
optimizer = torch.optim.Adam([noise], lr=0.1, weight_decay=1e-6)
s = SaveFeatures(model.layer1)
for i in range(0, 0):
	print(i, np.min(noise.data.numpy()), np.max(noise.data.numpy()))
	model(imgTensor)
	loss = -s.features[0, 5].mean()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if i % 10 == 0:
		cv2.imshow("noise", enlarge(noise[0][0].data.numpy()))
		cv2.imshow("noiseNorm", enlarge(normalize(noise[0][0].data.numpy())))
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
s.close()


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

