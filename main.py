import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def mse(imageA, imageB):
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	return err

while True:
    ret, cam = cap.read()
    cv2.imshow('Video Output', cam)

    imgFile = cv2.imread('test.jpg')
    resize_img = cv2.resize(imgFile, (640,480))
    cv2.imshow('image', imgFile)

    #   Compare both images
	# 	No longer using mse function, as it does not compare edges
    #print(mse(cam,resize_img))

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
