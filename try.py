import numpy as np
import cv2
from matplotlib import pyplot as plt

#cap = cv2.VideoCapture(0)
#k = cap.read()


#cap = cv2.VideoCapture(0)
img1 = cv2.imread('test.jpg',0)
#img2 = cv2.imshow(k,0)
img2 = cv2.imread('cam_img.jpg',0)


orb = cv2.ORB_create();

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
#kp2, des2 = orb.detectAndCompute(cap, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)

plt.imshow(img3),plt.show()
