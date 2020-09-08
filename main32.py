# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	#img = load_image('img2.jpg')
	img1=cv2.imread('img5.jpg',0)
	img1 = cv2.resize(img1,  (28, 28)) 
	img1.reshape((28,28)).astype('float32')
	print(img1.shape)
	batch = np.expand_dims(img1,axis=0)
	print(batch.shape) # (1, 28, 28)
	batch = np.expand_dims(batch,axis=3)
	print(batch.shape)
	batch=batch/255
	model = load_model('final_model.h5')
	#img1 = img1.reshape(img1.shape[0], 720, 1280, 1)
	#img=img/255
	# load model
	#img1 = img1.ravel()
	#img1 = np.invert(img1).ravel()
	#print(img1)
	#print(type(img))
	#print(type(img1))
	#print("hello")
	#print(len(img))
	#print("hello1")
	#print(len(img1))
	# predict the class
	digit = model.predict_classes(batch)
	print(digit[0])
 
# entry point, run the example
run_example()
