from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import (Input, Model, layers, losses, optimizers, metrics, utils, models)
# from os.path import exists as file_exists
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D

class Strategy(ABC):
    model: Model 
    image_shape = (224, 224, 3)    
    image_size = (224, 224) # type: ignore
    n_classes = 100

    @abstractmethod
    def loading_weights(self) -> None:
        pass
    
# class MobileNet(Strategy):
#     def training(self) -> str:
#         return f'MobileNet Model predicting '

class CNN(Strategy):
    def __init__(self) -> None:
        input_layer = Input(self.image_shape)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(input_layer)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.n_classes, name="final_dense", activation='softmax')(x)
        self.model = Model(input_layer, outputs)  # type: ignore
        self.loading_weights()  # type: ignore
        super().__init__()

    def loading_weights(self) -> None:
        batch_size = 64
        epochs = 1
        learning_rate = 1e-3 
        weights_file = '/home/skawy/side_projects/Sports-Image-Classification/models/weights/cnn_weights.h5'
        self.model.compile(optimizer = optimizers.Adam(learning_rate),   # type: ignore
                        loss = losses.categorical_crossentropy, 
                        metrics = ['accuracy'])

        self.model.load_weights(weights_file)  # type: ignore

        # history = model.fit(train_generator, validation_data= val_generator, epochs = EPOCHS)
    
    def  predict(self,image) -> float:
        return self.model.predict(image)


class MobileNet(Strategy):
    def __init__(self) -> None:
        base_model = MobileNetV2(include_top=False, input_shape=self.image_shape)
        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128))
        model.add(Dropout(0.1))
        model.add(Dense(128))
        model.add(Dropout(0.1))
        model.add(Dense(self.n_classes, activation = 'softmax'))

        self.model = model
        self.loading_weights()  # type: ignore
        super().__init__()

    def loading_weights(self) -> None:
        batch_size = 64
        epochs = 1
        learning_rate = 1e-3 
        weights_file = '/home/skawy/side_projects/Sports-Image-Classification/models/weights/mobilenet_weights1.h5'
        self.model.compile(optimizer = optimizers.Adam(learning_rate),   # type: ignore
                        loss = losses.categorical_crossentropy, 
                        metrics = ['accuracy'])

        self.model.load_weights(weights_file )  # type: ignore

        # history = model.fit(train_generator, validation_data= val_generator, epochs = EPOCHS)
    
    def  predict(self,image) -> float:
        return self.model.predict(image)

        
class CustomModel:
    strategy: Strategy

    def __init__(self, strategy: Strategy = None) -> None:  # type: ignore
        if strategy is not None:
            self.strategy = strategy
    
    def predict(self, image_path):
        image = cv2.imread(image_path)
        image = image.reshape(1,224,224,3)
        prediction = self.strategy.predict(image)  # type: ignore
        
        return np.argmax(prediction)
