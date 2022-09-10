# import keras 
# import numpy as np 

# from keras.applications import vgg16, inception_v3, mobilenet 

# #Load the VGG model 
# vgg_model = vgg16.VGG16(weights = 'imagenet') 

# #Load the Inception_V3 model 
# inception_model = inception_v3.InceptionV3(weights = 'imagenet') 

# print(vgg_model)

import os
from os.path import exists as file_exists
from statistics import mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2

import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras import (Input, Model, layers, losses, optimizers, metrics, utils, models)

sns.set_style('darkgrid')

DATA_PATH = '/home/skawy/side_projects/Sports-Image-Classification/kaggle_dataset'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VAL_PATH = os.path.join(DATA_PATH, 'valid')
TEST_PATH = os.path.join(DATA_PATH, 'test')

IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = len(os.listdir(TEST_PATH))

# HYPERPARAMETERS
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-3

data_generator = image.ImageDataGenerator(rescale = 1./255)
train_generator = data_generator.flow_from_directory(directory= TRAIN_PATH,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode= 'rgb',
                                                    class_mode= 'categorical',
                                                    batch_size= BATCH_SIZE)
val_generator = data_generator.flow_from_directory(directory= VAL_PATH,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode= 'rgb',
                                                    class_mode= 'categorical',
                                                    batch_size= BATCH_SIZE)
test_generator = data_generator.flow_from_directory(directory= TEST_PATH,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode= 'rgb',
                                                    class_mode= 'categorical',
                                                    batch_size= BATCH_SIZE,
                                                    shuffle= False)


def get_cnn_model(IMAGE_SHAPE):
    """
    creates and returns a CNN model.
    """
    # Define the tensors for the two input images
    input_layer = Input(IMAGE_SHAPE)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(NUM_CLASSES, name="final_dense", activation='softmax')(x)
    return Model(input_layer, outputs)

model = get_cnn_model(IMAGE_SHAPE)
print("type is ")
print(type(Model))
model.summary()

model.compile(optimizer = optimizers.Adam(LEARNING_RATE), 
                loss = losses.categorical_crossentropy, 
                metrics = ['accuracy'])

if(file_exists("cnn_weights.h5")):
    model.load_weights('cnn_weights.h5')
else:
    history = model.fit(train_generator, validation_data= val_generator, epochs = EPOCHS)


img_path = "/home/skawy/side_projects/Sports-Image-Classification/kaggle_dataset/train/wheelchair racing/002.jpg"
img = cv2.imread(img_path)
#print its shape
print('Image Dimensions :', img.shape)

img = img.reshape(1,224,224,3)
print('Image Dimensions :', img.shape)

prediction = model.predict(img)
print(np.argmax(prediction))

df = pd.read_csv('/home/skawy/side_projects/Sports-Image-Classification/class_dict.csv')
# print(df.to_string())

sports = df['class'] 

print(sports[np.argmax(prediction)])