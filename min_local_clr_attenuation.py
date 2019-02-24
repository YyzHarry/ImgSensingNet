# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

SCALE_LOCAL_CLR_ATTEN = 15

# trained hyperparam from paper
# Q. Zhu et al., ``A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior''
# https://ieeexplore.ieee.org/document/7128396
SIGMA = 0.041337
THETA0 = 0.121779
THETA1 = 0.959710
THETA2 = -0.780245


def Min_Local_Clr_Atten(image):
    height, width, _ = image.shape
    
    # normalize AFTER TRANS
    hsv_img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
    img_s = hsv_img[:, :, 1] / 255.
    img_v = hsv_img[:, :, 2] / 255.

    sigmaMat = np.random.normal(0, SIGMA, (height, width))
    depth_image = THETA0 + THETA1 * img_v + THETA2 * img_s + sigmaMat

    # default SCALE_LOCAL_CLR_ATTEN = 15
    pad_size = int(SCALE_LOCAL_CLR_ATTEN/2)
    # default padding val = 1
    padimage = np.pad(depth_image, ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=(1, 1))

    clr_atten_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+SCALE_LOCAL_CLR_ATTEN), j:(j+SCALE_LOCAL_CLR_ATTEN)]
            clr_atten_image[i, j] = np.min(patch)
    return clr_atten_image


if __name__ == '__main__':

    fn = 'results/forest.jpg'
    src = cv2.imread(fn)

    # normalize AFTER TRANS
    # need to preprocess in the main file
    I = src.astype('float64')
    
    # Patch_Size = 15*15
    # Get local-color-attenuation
    clr_atten = Min_Local_Clr_Atten(I)
    
    # Adaptive window size (for large input img)
    # cv2.namedWindow('Min_Local_Clr_Attenuation', cv2.WINDOW_NORMAL)
    # cv2.imshow('Min_Local_Clr_Attenuation', clr_atten)

    cv2.imwrite('results/forest_clratten.jpg', clr_atten * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
