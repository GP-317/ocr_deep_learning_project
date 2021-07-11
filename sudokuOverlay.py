##### Modules Imported

from numpy.matrixlib.defmatrix import matrix
from sudokuUtilities import *
import os
import sudokuSolver





################## Ressources ##################
pathImage = "ressources/sud1.jpg"
height = 450 # Needs to be divisible by 9
width = 450  # Needs to be divisible by 9
model = initializeModel()
################################################


##### Image preparation ###
img = cv2.imread(pathImage)
# Resize the image to make it an actual square
# as it will be easier to work on
img = cv2.resize(img, (width, height))
# Blank image used while coding for test and debug operation
imgBlank = np.zeros((height, width, 3), np.uint8)
imgThreshold = preProcess(img)


##### Separation of images' elements from the sudoku grid
# Copy of initial image to get the basis on which we will find
# each outlines (defining a contour) the reference has
imgOutlines = img.copy()
imgBigOutlines = img.copy()
# As said by the function, it will find every contours 
# existing in the reference
outlines, hierarchy = cv2.findContours(imgThreshold, 
cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# As said by the function, it will draw each contours
# found previously
cv2.drawContours(imgOutlines, outlines, -1, (0, 255, 0), 3)


##### Finding the biggest contour and using it as the sudoku grid
# Find the biggest contour based on a sudokoUtilities' function
biggest, maxArea = biggestContour(outlines)
# the 'biggest' variable contains the coordinates of the biggest contour
if biggest.size != 0:
    # The reorder function will set every points in the biggest 
    # array in a specific order to prevent a mismatch
    biggest = reorder(biggest)
    # As said, the function will draw the biggest contour
    cv2.drawContours(imgBigOutlines, biggest, -1, (0, 255, 0), 10)
    # Next two variables are set to prepare points for warp
    point1 = np.float32(biggest)
    point2 = np.float32([[0, 0],  [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(point1, point2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_RGB2GRAY)


##### Splitting the image and then proceed to find each digit 
# thanks to the trained model
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
numbers = getPrediction(boxes, model)

imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(0, 0, 255))
numbers = np.asarray(numbers)
posArray = np.where(numbers > 0, 0, 1)

imgDetectedDigits = drawGrid(imgDetectedDigits)


##### Step where the program will find the solution of the board, 
# which needs to use an external sudoku solver
# the grid is splitted in 9 arrays
board = np.array_split(numbers,9)
try:
    sudokuSolver.solve(board)
except:
    pass

flatlist = []
for sublist in board:
    for item in sublist:
        flatlist.append(item)
solvedBoard = flatlist*posArray
imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedBoard)


##### Overlaying step, will overlay the solution grid over the sudoku grid given at the start

point2 = np.float32(biggest)
point1 = np.float32([[0, 0],  [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(point1, point2)
imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width, height))
inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)

imgSolvedDigits = drawGrid(imgSolvedDigits)


imageArray = ([img, imgThreshold, imgOutlines, imgBigOutlines],
            [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
stackedImages = stackImages(imageArray, 0.7)
cv2.imshow('Overlay and Solver Presentation', stackedImages)


cv2.waitKey(0)