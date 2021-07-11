# -*- coding: utf8 -*-

"""
Training of a convolutional network with images stored in 
the "dataset" folder. 

"""

__author__ =  'Thierry BROUARD', 'Geoffrey PRIVARD'
__version__=  '0.1.1'

# Initial code from Thierry BROUARD
# Updated version from Geoffrey PRIVARD
#
# Optical Character Recognition with hand-written numbers instead 
# of hand-written letters in order to implement a sudoku solver 
# using the trained convolutional network made with this file



import argparse
from imutils import paths
import os
from matplotlib import image
from numpy.lib.npyio import load
from skimage import io 
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import model_lenet as mdl
import info as info

# global vars
target_shape = (28, 28, 1) # rows, cols of final image
model_name = "./models/lenet" # name of the model for saving


def loadPicture(imageName):

	print( "[INFO] reading ", imageName, "..." )
	# load image
	imgOrig = io.imread( imageName )
	imgOrig = np.reshape( imgOrig, (imgOrig.shape[0], imgOrig.shape[1], 1) )
	print( "[INFO] processing image ..." )
	# processing : resize the original image, ratio preserving, to fit the target
	# size along one or two dimensions
	scalex = target_shape[1] / imgOrig.shape[1]
	scaley = target_shape[0] / imgOrig.shape[0]
	scale = min (scalex, scaley)

	# rescaling
	imgResized = rescale(imgOrig, scale, anti_aliasing=True)

	# embedding in blank image, center of resized image fit to center of target image
	imgBlank = np.ones(target_shape) * imgResized.max()
	imgResizedWidth  = imgResized.shape[1]
	imgResizedHeight = imgResized.shape[0]
	# offset to store data in the rescaled image
	offsetx = (imgBlank.shape[1] - imgResizedWidth) / 2
	offsety = (imgBlank.shape[0] - imgResizedHeight)/ 2
	# copy values at the right place
	imgBlank[int(offsety):int(offsety+imgResizedHeight), int(offsetx):int(offsetx+imgResizedWidth)] = imgResized
	# values between 0 and 1
	#imgBlank = imgBlank / imgBlank.max()
	return imgBlank

def plot_images_sample(X, Y, titre):

    fig = plt.figure(1)
    rand_indicies = np.random.randint(len(X), size=25)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        index = rand_indicies[i]
        plt.imshow(np.squeeze(X[index]), cmap=plt.cm.binary)
        plt.xlabel(Y[index])
    
    fig.suptitle(titre)
    plt.show()





# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=50,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading images...")

path = args["dataset"]
imagePaths = os.listdir(path)
noOfClasses = len(imagePaths)
data = []
labels = []

# # loop over the image paths
# for imagePath in imagePaths:
# 	# extract the class label from the filename
# 	label = imagePath.split(os.path.sep)[-2].split("-")[-2]
# 	# load the image and resize it to be a fixed 32x32 pixels,
# 	# preserving aspect ratio, and values between 0..1
# 	image = loadPicture(imagePath)
# 	# update the data and labels lists, respectively
# 	data.append(image)
# 	labels.append(label)

# print(noOfClasses)
# print(imagePaths)

# loop over the range of classes
for i in range(0, noOfClasses):
    
    # extract each class folder and set it as an images list
    datasetPicList = os.listdir(path+str(i))
   
    # loop over the image paths given previously
    for j in datasetPicList:
        
        # usage of loadPicture class to load and 
        # resize the image if needed
        image = loadPicture(path+str(i)+'/'+str(j))
        
        # updates the data and labels(//class) list respectively
        data.append(image)
        labels.append(i)

# print(labels)

plot_images_sample(data, labels, "Some samples of pictures to learn")

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") #/ 255.0
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.30, shuffle=True)
# construct the training image generator for data augmentation
#aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#	horizontal_flip=True, fill_mode="nearest")


aug = ImageDataGenerator(
    # featurewise_center=False,
    # samplewise_center=False,
    # featurewise_std_normalization=False,
    # samplewise_std_normalization=False,
    # zca_whitening=False,
    # zca_epsilon=1e-06,
    rotation_range=10, # 0
    width_shift_range=0.1, # 0.0
    height_shift_range=0.1, # 0.0
    # brightness_range=None,
    shear_range=0.1, # 0.0
    zoom_range=0.2, #0.0
    # channel_shift_range=0.0,
    # fill_mode='nearest',
    # cval=0.0,
    # horizontal_flip=False,
    # vertical_flip=False,
    # rescale=None,
    # preprocessing_function=None,
    # data_format=None,
    # validation_split=0.0,
    # dtype=None
    )


print('[INFO] - Creating model...')
model = mdl.get_model()

model.summary()

print('[INFO] - Training model...')
# training with data augmentation... 
history = model.fit(x=aug.flow(trainX, trainY, batch_size=32, ), #save_to_dir="./aug/"),
          steps_per_epoch=len(trainX) // 32,
          validation_data=(testX, testY),
          epochs=args["epochs"],
          verbose = 1
          )
# training without data augmentation... 

# history = model.fit(
#           trainX,
#           trainY,
#           batch_size = 32,
#           epochs = args["epochs"],
#           verbose = 1,
#           validation_data=(testX, testY)
#           )

print('[INFO] - Saving model...')
mdl.save(model, model_name)

info.plot_performance(history)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=imagePaths))
#print(confusion_matrix(testY.argmax(axis=1),
#	predictions.argmax(axis=1), labels=lb.classes_))

# les données sont présentées dans testX, le résultat est dans testY.argmax(axis=1)
# qui est l'indice de la classe reconnue (espace python), qui peut être décodée
# via le tableau lb.classes_
#
# Update: target_names n'est plus égal à lb.classes mais à imagePaths
# car le résultat différait entre les deux méthodes. imagePaths renvoie 
# la liste des dossiers présents dans .\dataset\, ce qui correspond au 
# résultat attendu par le model

# python3 ./train.py -d ./trainingSet/ -e 20 




