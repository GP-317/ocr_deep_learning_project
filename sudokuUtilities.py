### Modules Imported
# from model_lenet import compile_model
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import models



##### Pre-Processing of image
def preProcess(img):
    # Converts the image color space to a gray color space
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurs the image using a gaussian filter
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # Applies an adaptative threshold to the image
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold


##### Function to find the biggest contour (assuming that is the sudoku grid)
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


##### Function that reorders points for Warp Perspective
def reorder(pointsReorder):
    pointsReorder = pointsReorder.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = pointsReorder.sum(1)
    myPointsNew[0] = pointsReorder[np.argmin(add)]
    myPointsNew[3] = pointsReorder[np.argmax(add)]
    diff = np.diff(pointsReorder, axis=1)
    myPointsNew[1] = pointsReorder[np.argmin(diff)]
    myPointsNew[2] = pointsReorder[np.argmax(diff)]
    return myPointsNew


##### Function to split the images in 81 images corresponding to the 81 boxes in the grid
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


##### Function used to initialize the trained model
def initializeModel():
    model = load_model('model_trained.h5')

    # #Reading the model from JSON file
    # with open('models/lenet_struct.json', 'r') as json_file:
    #     json_savedModel= json_file.read()

    # #load the model architecture 
    # model = models.model_from_json(json_savedModel)

    # # load the weights
    # model.load_weights('models/lenet_weights.h5')

    # # compile again
    # model = compile_model(model)
    return model 


##### Function to get the predictions on every images rendered through the split function
def getPrediction(boxes,model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28)) #28
        img = img / 255
        img = img.reshape(1, 28, 28, 1) #28
        ## GET PREDICTION
        predictions = model.predict(img)
        # classIndex = model.predict_classes(img)       DEPRECATED
        classIndex = np.argmax(model.predict(img), axis=-1)
        probabilityValue = np.amax(predictions)

        # print(classIndex, probabilityValue)   
        # For testing purposes, prints in the console the probability linked to the class indexed

        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


##### Function to display the numbers on an image
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


##### Function that will draw a grid, according to the warp perspective part
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img





##### Stack images in one window for presentation purposes 
# and step-by-step helper in coding
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver