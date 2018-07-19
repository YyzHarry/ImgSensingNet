# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np


# ========================================== Hue Disparity Calculation
def Hue_Disp(image):
    height, width, channel = image.shape
    # semi-inverse calculation
    sinv_image = image.copy()
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                if(sinv_image[i,j,k] < 0.5):
                    sinv_image[i,j,k] = 1 - sinv_image[i,j,k]

    # convert into HLS color space
    hls_image = np.uint8(image * 255.0)
    hls_sinv_image = np.uint8(sinv_image * 255.0)
    hls_image = cv2.cvtColor(hls_image, cv2.COLOR_BGR2HLS)
    hls_sinv_image = cv2.cvtColor(hls_sinv_image, cv2.COLOR_BGR2HLS)

    # hue calculation
    hue = hls_image[:,:,0]
    sinv_hue = hls_sinv_image[:,:,0]
    hue = hue / 180.0
    sinv_hue = sinv_hue / 180.0
    hue_image = np.abs(hue - sinv_hue)
    return hue_image


# ========================================== Main
if __name__ == '__main__':
    # Read in by streaming? or one by one?
    fn = 'Results/8.png'
    
    src = cv2.imread(fn)
    # print(src)
    
    # Image normalization
    I = src.astype('float64') / 255
    
    # Get hue-disparity
    hue = Hue_Disp(I)
    
    # Adaptive window size (for large input img)
    cv2.namedWindow('Hue_Disparity', cv2.WINDOW_NORMAL)
    cv2.imshow('Hue_Disparity', hue)

    # Need to rescale to [0,255]
    cv2.imwrite('Results/hue_disp.jpg', hue * 255)
    
    cv2.waitKey(0)
