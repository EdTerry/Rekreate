import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

def mse(imageA, imageB):
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	return err

while True:
    ret, frame = cap.read()

    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    # create a red HSV colour boundary and
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Display an original image
    cv2.imshow('Original',frame)

    # finds edges in the input image image and
    # marks them in the output map edges
    cam_edges = cv2.Canny(frame,100,200)

    # Display edges in a frame
    cv2.imshow('Edges',cam_edges)

    imgFile = cv2.imread('test.jpg')
    imgFile_edges = cv2.Canny(imgFile,640,480)

    resize_img = cv2.resize(imgFile_edges, (640,480))

    cv2.imshow('image', resize_img)

    #   Compare both images
	# 	No longer using mse function, as it does not compare edges
    print(mse(cam_edges,resize_img))

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
