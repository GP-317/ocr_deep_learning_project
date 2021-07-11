# -*- coding: utf8 -*-

"""
Definition of a CNN (LeNet family), & loading + saving model.

"""

__author__ =  'Thierry BROUARD'
__version__=  '0.1'

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.python.keras.layers.core import Dropout

def get_model():
    # LeNet 
    model = models.Sequential([
        layers.Convolution2D(filters = 60, kernel_size = (5, 5), input_shape = (28, 28, 1)),
        layers.Activation(activation = "relu"),
        layers.MaxPooling2D(pool_size = (2, 2), strides =  (2, 2)),
        layers.Convolution2D(filters = 30, kernel_size = (3, 3)),
        layers.Activation(activation = "relu"),
        layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(500),
        layers.Activation(activation = "relu"),
        layers.Dropout(0.5),
        layers.Dense(10), # nb of output classes
        layers.Activation("softmax")
    ])

    return compile_model(model)



def compile_model(model):
	model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	return model 



def save(model, mdlname):
    # serialize model to json
    json_model = model.to_json()#save the model architecture to JSON file
    with open(mdlname+'_struct.json', 'w') as json_file:
        json_file.write(json_model)

    #saving the weights of the model
    model.save_weights(mdlname+'_weights.h5')

    model.save('myModel.h5')



def load(mdlname):
    #Reading the model from JSON file
    with open(mdlname+'_struct.json', 'r') as json_file:
        json_savedModel= json_file.read()

    #load the model architecture 
    model = models.model_from_json(json_savedModel)

    # load the weights
    model.load_weights(mdlname+'_weights.h5')

    # compile again
    model = compile_model(model)

    return model 


