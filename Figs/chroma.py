# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np


# ========================================== Chroma Calculation
def Chroma(image):
    # convert into LAB color space
    lab_image = np.uint8(image * 255.0)
    lab_image = cv2.cvtColor(lab_image, cv2.COLOR_BGR2LAB)
    a_image = lab_image[:,:,1]
    b_image = lab_image[:,:,2]
    a_image = a_image / 255.0
    b_image = b_image / 255.0
    # chroma calcultion
    ch_image = np.sqrt(a_image**2 + b_image**2)
    return ch_image


# ========================================== Main
if __name__ == '__main__':
    # Read in by streaming? or one by one?
    fn = 'Results/8.png'
    
    src = cv2.imread(fn)
    # print(src)
    
    # Image normalization
    I = src.astype('float64') / 255
    
    # Get chroma
    ch = Chroma(I)
    
    # Adaptive window size (for large input img)
    cv2.namedWindow('Chroma', cv2.WINDOW_NORMAL)
    cv2.imshow('Chroma', ch)

    # Need to rescale to [0,255]
    cv2.imwrite('Results/chroma.jpg', ch * 255)
    
    cv2.waitKey(0)
