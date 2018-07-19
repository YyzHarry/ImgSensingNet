# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

scale_local_sat = 10

# ========================================== Exception when Dividing Zero
def exc_div_zero(sat_min, sat_max):
    if(sat_max != 0):
        return (sat_min*1.0) / sat_max
    else:
        return 1


# ========================================== Max Local Saturation Calculation
def Max_Local_Sat(image):
    height, width, __ = image.shape
    sat_image = np.zeros((height, width))
    # 1*1 saturation calculate
    for i in range(height):
        for j in range(width):
            sat_image[i,j] = 1 - exc_div_zero(np.min(image[i,j,:]), np.max(image[i,j,:]))
    # scale_local_sat = 10, a hyperparam
    pad_size = int(scale_local_sat/2)
    # padding default value = 0
    padimage = np.pad(sat_image, ((pad_size,pad_size), (pad_size,pad_size)), 'constant')

    sat_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+scale_local_sat), j:(j+scale_local_sat)]
            sat_image[i,j] = np.max(patch)
    return sat_image


# ========================================== Main
if __name__ == '__main__':
    # Read in by streaming? or one by one?
    fn = 'Results/8.png'
    
    src = cv2.imread(fn)
    # print(src)
    
    # Image normalization
    I = src.astype('float64') / 255
    
    # Patch_Size = 10*10
    # Get local-saturation
    saturation = Max_Local_Sat(I)
    
    # Adaptive window size (for large input img)
    cv2.namedWindow('Max_Local_Saturation', cv2.WINDOW_NORMAL)
    cv2.imshow('Max_Local_Saturation', saturation)

    # Need to rescale to [0,255]
    cv2.imwrite('Results/max_local_sat.jpg', saturation * 255)
    
    cv2.waitKey(0)
