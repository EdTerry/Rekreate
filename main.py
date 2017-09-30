import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

while True:
    ret, cam = cap.read()
    cv2.imshow('Video Output', cam)

    imgFile = cv2.imread('test.jpg')
    resize_img = cv2.resize(imgFile, (480,640))
    cv2.imshow('image', imgFile)

    #   Compare both images
    print(mse(cam,imgFile))

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
