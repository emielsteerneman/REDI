import numpy as np
import cv2

COLOURS = [(255, 127, 0), (127, 0, 255), (0, 255, 127), (127, 255, 0), (255, 0, 127), (0, 127, 255), (0, 0, 204), (0, 204, 0)]

def renderBarChart(values, BAR_WIDTH=2, PADDING=2, FACTOR=100, GRAPH_HEIGHT=200, COLOUR=(255, 255, 255)):
	if len(values.shape) != 1:
		print("BRUH!", values.shape)

	NVALUES = values.shape[0]
	GRAPH_WIDTH = NVALUES * (BAR_WIDTH+PADDING) - PADDING
	chart = np.ones((GRAPH_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8) * 20
	x = 0
	for v in values:
		y = int(v*FACTOR)
		cv2.rectangle(chart, (x, GRAPH_HEIGHT//2), (x + BAR_WIDTH, GRAPH_HEIGHT//2-y), COLOUR, -1)
		x += BAR_WIDTH + PADDING
	return chart

def render2dBarCharts(values, BAR_WIDTH=2, PADDING=2, FACTOR=100, GRAPH_HEIGHT=200, COLOUR=None, CLASS_PADDING=10):
	if len(values.shape) != 2:
		print("BRUH!**2", values.shape)
	## Render all bar charts
	renders = []
	for iValue, value in enumerate(values):
		colour = COLOUR if COLOUR is not None else COLOURS[iValue%len(COLOURS)] # Grab a colour if not provided
		render = renderBarChart(value, BAR_WIDTH, PADDING, FACTOR, GRAPH_HEIGHT, colour) # Render bar chart
		renders.append(render) # Add to list of rendered bar charts
	## Calculate some values about width and height etc
	RENDER_HEIGHT, RENDER_WIDTH, _ = renders[0].shape
	GRAPH_WIDTH = values.shape[0] * (RENDER_WIDTH+CLASS_PADDING) - CLASS_PADDING
	## Create image
	chart = np.ones((RENDER_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8) * 20
	## Fill image with all the renders
	for iRender, render in enumerate(renders):
		x = iRender * (RENDER_WIDTH+CLASS_PADDING)
		chart[0:RENDER_HEIGHT, x:x+RENDER_WIDTH, :] = render
	
	return chart

def render3dBarCharts(values, BAR_WIDTH=2, PADDING=2, FACTOR=100, GRAPH_HEIGHT=200):
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

def renderConvLayer(values, PADDING=2):
	if len(values.shape) == 4 and values.shape[0] == 1:
		values = np.squeeze(values, axis=0) # remove first dimension

	if len(values.shape) != 3:
		print("Wrong input shape for renderConvLayer")

	IMAGE_SIZE = 80

	IMAGE_HEIGHT = values.shape[0] * (IMAGE_SIZE + PADDING) - PADDING

	render = np.ones((IMAGE_HEIGHT, IMAGE_SIZE, 3), dtype=np.uint8) * 20
	for iImage, image in enumerate(values):
		## Upscale to IMAGE_SIZExIMAGE_SIZE, divide by 5 to get values to [0, 1] (seemed a nice value)
		factor = IMAGE_SIZE // image.shape[0]
		
		# img = np.kron(image, np.ones((factor, factor))) / 5.0
		img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_NEAREST) 
		img = img * (255 / 5)

		y1 = iImage * (IMAGE_SIZE + PADDING)
		y2 = y1 + IMAGE_SIZE
		render[y1:y2, :, 0] = img
		render[y1:y2, :, 1] = img
		render[y1:y2, :, 2] = img

	return render