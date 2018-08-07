# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

scale_local_clr_atten = 15

# ========================================== Min Local Color Attenuation Calculation
def Min_Local_Clr_Atten(image):
    height, width, __ = image.shape
    
    # WARNING: normalize AFTER TRANS!!!
    # Current: After
    hsv_img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
    img_s = hsv_img[:,:,1] / 255.0
    img_v = hsv_img[:,:,2] / 255.0

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (height, width))
    depth_image = 0.121779 + 0.959710 * img_v - 0.780245 * img_s + sigmaMat

    # scale_local_clr_atten = 10, a hyperparam
    pad_size = int(scale_local_clr_atten/2)
    # padding default value = 1
    padimage = np.pad(depth_image, ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values = (1, 1))

    clr_atten_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+scale_local_clr_atten), j:(j+scale_local_clr_atten)]
            clr_atten_image[i,j] = np.min(patch)
    return clr_atten_image


# ========================================== Main
if __name__ == '__main__':
    # Read in by streaming? or one by one?
    fn = 'Results/8.png'
    
    src = cv2.imread(fn)
    # print(src)

    # WARNING: normalize AFTER TRANS!!! (the only one)
    # Current: After, need to preprocess in the main.py
    I = src.astype('float64')
    
    # Patch_Size = 15*15
    # Get local-color-attenuation
    clr_atten = Min_Local_Clr_Atten(I)
    
    # Adaptive window size (for large input img)
    cv2.namedWindow('Min_Local_Clr_Attenuation', cv2.WINDOW_NORMAL)
    cv2.imshow('Min_Local_Clr_Attenuation', clr_atten)

    # Need to rescale to [0,255]
    cv2.imwrite('Results/min_local_clr_atten.jpg', clr_atten * 255)
    
    cv2.waitKey(0)
