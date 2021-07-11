# -*- coding: utf8 -*-

"""
Training of a convolutional network with images stored in 
the "dataset" folder. 
Type of data trained: numeric digit numbers

"""



import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

############# Ressources ##############
path = 'myDataset'
testRatio = 0.2
validationRatio = 0.1
imageDimensions = (28, 28, 3) #28

batchSizeVal= 50
epochsVal = 20
stepsPerEpochVal = 1000

#######################################

images = []

dataset = os.listdir(path)
classNo = []
print("Total of labels detected :", len(dataset))
noOfClasses = len(dataset)

print("Importing Labels .....")
for i in range(0, noOfClasses):
    datasetPicList = os.listdir(path+"/"+str(i))
    for j in datasetPicList:
        currentImg = cv2.imread(path+"/"+str(i)+"/"+str(j))
        currentImg = cv2.resize(currentImg, (28,28))
        images.append(currentImg)
        classNo.append(i)
    print(i, end=" ")
print(" ")


images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
# print(labelNumber.shape)

### Partitionning and Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(
                                    images,
                                    classNo,
                                    test_size=testRatio)

X_train,X_validation,Y_train,Y_validation = train_test_split(
                                            X_train,
                                            Y_train,
                                            test_size=validationRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

noOfSamples = []
for x in range(0, noOfClasses):
    print(len(np.where(Y_train==x)[0]))
    noOfSamples.append(len(np.where(Y_train==x)[0]))
print(noOfSamples)


### Plotting data

plt.figure(figsize=(10,5))
plt.bar(range(0, noOfClasses), noOfSamples)
plt.title("Number of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()



def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[46])
# img = cv2.resize(img,(150,150))
# cv2.imshow("Pre-Processed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)


dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

dataGen.fit(X_train)

Y_train = to_categorical(Y_train, noOfClasses)
Y_test = to_categorical(Y_test, noOfClasses)
Y_validation = to_categorical(Y_validation, noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,
                        sizeOfFilter1,
                        input_shape=(
                            imageDimensions[0],
                            imageDimensions[1],
                            1),
                        activation="relu"
                    )))
    
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation="relu")))

    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation="relu")))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation="relu")))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(noOfClasses, activation = "softmax"))
    model.compile(Adam(learning_rate=0.001), loss=losses.CategoricalCrossentropy(from_logits=True), 
                    metrics = ['accuracy'])
    return model

model = myModel()

print(model.summary())


#### STARTING THE TRAINING PROCESS
history = model.fit_generator(dataGen.flow(X_train,Y_train, batch_size = batchSizeVal),
                                 steps_per_epoch=len(X_train)//batchSizeVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,Y_validation),
                                 shuffle=1)

#### PLOT THE RESULTS  
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,Y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

#### SAVE THE TRAINED MODEL 
model.save("model_trained.h5")