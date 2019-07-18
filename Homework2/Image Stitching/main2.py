import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import random

# 此代码完成的是图像拼接的功能

# 读取图片数据
img1 = cv2.imread('buildings_part1.jpg')  # queryImage
img2 = cv2.imread('buildings_part2.jpg') # trainImage
print('Image 1 shape:', img1.shape)
print('Image 2 shape:', img2.shape)

# 获得图片对应的灰度图像。
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Define SIFT detector and initialize, apply SIFT detector and find keypoints, compute descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print('Image 1: {} keypoints found.'.format(len(kp1)))
print('Image 2: {} keypoints found.'.format(len(kp2)))

#Plot image with keypoints marked with circle and orientation arrow
img1_kp = cv2.drawKeypoints(img1, kp1, outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Define algorithm to match sift descriptors, find inliers
bf = cv2.BFMatcher_create(crossCheck=True)
matches = bf.match(des1, des2)
print(len(matches),' matches found.')

pts1 = []
pts2 = []
for mat in matches:
    pts1.append(kp1[mat.queryIdx].pt)
    pts2.append(kp2[mat.trainIdx].pt)
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

# Compute homography matrix
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
print('Computed homography matrix: \n', H)

# 生成拼接图
pano_size = (img1.shape[0]+img2.shape[0], img1.shape[1]+img2.shape[1])
img_pano = cv2.warpPerspective(img2, H, pano_size)
img_pano[0:img1.shape[0], 0:img1.shape[1], :] = img1
cv2.imshow("Stitching", img_pano) #blend/cut/fade
cv2.waitKey(0)
cv2.destroyAllWindows()
