import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

img_gray = cv2.imread('D:/Code/Test/Snorlax.jpg', 0)
cv2.imshow('lenna', img_gray)