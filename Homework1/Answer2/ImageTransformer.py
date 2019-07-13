#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import random
import numpy as np


class ImageTransformer:
    empCount = 0

    @staticmethod
    def crop(img, h_start, h_end, w_start, w_end):
        img_crop = img[h_start:h_end, w_start:w_end]
        return img_crop

    @staticmethod
    def random_light_color(img):
        # brightness
        B, G, R = cv2.split(img)

        b_rand = random.randint(-50, 50)
        if b_rand == 0:
            pass
        elif b_rand > 0:
            lim = 255 - b_rand
            B[B > lim] = 255
            B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
        elif b_rand < 0:
            lim = 0 - b_rand
            B[B < lim] = 0
            B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

        g_rand = random.randint(-50, 50)
        if g_rand == 0:
            pass
        elif g_rand > 0:
            lim = 255 - g_rand
            G[G > lim] = 255
            G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
        elif g_rand < 0:
            lim = 0 - g_rand
            G[G < lim] = 0
            G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

        r_rand = random.randint(-50, 50)
        if r_rand == 0:
            pass
        elif r_rand > 0:
            lim = 255 - r_rand
            R[R > lim] = 255
            R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
        elif r_rand < 0:
            lim = 0 - r_rand
            R[R < lim] = 0
            R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

        img_merge = cv2.merge((B, G, R))
        # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img_merge

    @staticmethod
    def rotate(img, m, n, angle, scale):
        # rotation
        matrix = cv2.getRotationMatrix2D((m, n), angle, scale)  # center, angle, scale
        img_rotate = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        return img_rotate

    @staticmethod
    # perspective transform
    def random_warp(img):
        height, width, channels = img.shape

        # warp:
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        return M_warp, img_warp
