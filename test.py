import keras 
import numpy as np 

from keras.applications import vgg16, inception_v3, mobilenet 

#Load the VGG model 
vgg_model = vgg16.VGG16(weights = 'imagenet') 

#Load the Inception_V3 model 
inception_model = inception_v3.InceptionV3(weights = 'imagenet') 

print(vgg_model)