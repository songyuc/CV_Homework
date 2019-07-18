import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import random

#img1 = cv.imread('p1.jpg',0)  # trainImage
img1 = cv.imread('p1.jpg')  # queryImage
img2 = cv.imread('p0.jpg') # trainImage
print('Image 1 shape:', img1.shape)
print('Image 2 shape:', img2.shape)
'''
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1)
a.set_title('img1')
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img2)
a.set_title('img2')
'''

img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#Define SIFT detector and initialize, apply SIFT detector and find keypoints, compute descriptors
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print('Image 1: {} keypoints found.'.format(len(kp1)))
print('Image 2: {} keypoints found.'.format(len(kp2)))

#Plot image with keypoints marked with circle and orientation arrow
img1_kp = cv.drawKeypoints(img1, kp1, outImage=np.array([]),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv.drawKeypoints(img2, kp2, outImage=np.array([]),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
'''
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1_kp)
a.set_title('img1_kp')
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img2_kp)
a.set_title('img2_kp')
'''

#Define algorithm to match sift descriptors, find inliers
bf = cv.BFMatcher_create(crossCheck=True)
matches = bf.match(des1,des2)
print(len(matches),' matches found.')

pts1 = []
pts2 = []
for mat in matches:
    pts1.append(kp1[mat.queryIdx].pt)
    pts2.append(kp2[mat.trainIdx].pt)
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

# Compute homography matrix
H, _ = cv.findHomography(pts2, pts1, cv.RANSAC)
print('Computed homography matrix: \n', H)

#Create panorama
pano_size = (img1.shape[0]+img2.shape[0], img1.shape[1]+img2.shape[1])
img_pano = cv.warpPerspective(img2, H, pano_size)
img_pano[0:img1.shape[0], 0:img1.shape[1], :] = img1
cv.imshow("Panorama", img_pano) #blend/cut/fade
cv.waitKey(0)
cv.destroyAllWindows()
