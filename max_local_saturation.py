# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

SCALE_LOCAL_SAT = 10


def exc_div_zero(sat_min, sat_max):
    if(sat_max):
        return (sat_min*1.) / sat_max
    else:
        return 1


def Max_Local_Sat(image):
    height, width, __ = image.shape
    sat_image = np.zeros((height, width))
    # 1*1 saturation calculate
    for i in range(height):
        for j in range(width):
            sat_image[i, j] = 1 - exc_div_zero(np.min(image[i, j, :]), np.max(image[i, j, :]))
    # default SCALE_LOCAL_SAT = 10
    pad_size = int(SCALE_LOCAL_SAT/2)
    # default padding val = 0
    padimage = np.pad(sat_image, ((pad_size,pad_size), (pad_size,pad_size)), 'constant')

    sat_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+SCALE_LOCAL_SAT), j:(j+SCALE_LOCAL_SAT)]
            sat_image[i, j] = np.max(patch)
    return sat_image


if __name__ == '__main__':

    fn = 'results/forest.jpg'
    src = cv2.imread(fn)
    
    # Image normalization
    I = src.astype('float64') / 255.
    
    # Patch_Size = 10*10
    # Get local-saturation
    saturation = Max_Local_Sat(I)
    
    # Adaptive window size (for large input img)
    # cv2.namedWindow('Max_Local_Saturation', cv2.WINDOW_NORMAL)
    # cv2.imshow('Max_Local_Saturation', saturation)

    cv2.imwrite('results/forest_saturat.jpg', saturation * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
