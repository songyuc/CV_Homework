import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 自定义类
import ImageTransformer


# “image crop, color shift, rotation and perspective transform”
# 此次图像处理的作业共有四项图像裁剪、颜色转换、图像旋转和投影变换，要求写在一个类中。
GROUP_SIZE = 6


# 打开一张图片资源。
img_gray = cv2.imread('D:/Code/Test/Wish.jpg', 0)
imageTransformer = ImageTransformer.ImageTransformer()

# 共有四项任务 首先进行第一项
# 1. 图像裁剪
# 目标是前六张图片
task_id = 1
i = 0
while i < GROUP_SIZE:
    img = cv2.imread('./pokemon/pokemon'+str(i+1)+'.jpeg')
    h_half = int(img.shape[0]/2)
    w_half = int(img.shape[1]/2)
    img_crop = imageTransformer.crop(img, 0, h_half-1, 0, w_half-1)
    cv2.imwrite('./output/pokemon'+str(i+1)+'_crop.jpeg', img_crop)
    # 利用矩阵操作进行图片剪切
    i = i+1


# 2. 颜色转换
task_id = 2
i = 0
while i < GROUP_SIZE:
    origin_name = 'pokemon'+str((task_id-1)*6+i+1)
    img = cv2.imread('./pokemon/'+origin_name+'.jpeg')
    img_random_color_change = imageTransformer.random_light_color(img)
    cv2.imwrite('./output/'+origin_name+'_random_color_change.jpeg', img_random_color_change)
    i=i+1
# 完成

# 3. 图片旋转
task_id = 3
i = 0
while i < GROUP_SIZE:
    origin_name = 'pokemon'+str((task_id-1)*6+i+1)
    img = cv2.imread('./pokemon/'+origin_name+'.jpeg')
    img_rotate = imageTransformer.rotate(img, img.shape[1] / 2, img.shape[0] / 2, 30, 0.5)
    cv2.imwrite('./output/'+origin_name+'_rotate.jpeg', img_rotate)
    i = i+1
# 完成

# 4. 投影变换
task_id = 4
i = 0
while i < GROUP_SIZE:
    origin_name = 'pokemon'+str((task_id-1)*6+i+1)
    img = cv2.imread('./pokemon/'+origin_name+'.jpeg')
    M_warp, img_warp = imageTransformer.random_warp(img)
    cv2.imwrite('./output/'+origin_name+'_warp.jpeg', img_warp)
    i = i+1
# 完成

print('Processing finished!')
