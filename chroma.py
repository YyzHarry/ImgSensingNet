# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np


def Chroma(image):
    # convert into LAB color space
    lab_image = np.uint8(image * 255.)
    lab_image = cv2.cvtColor(lab_image, cv2.COLOR_BGR2LAB)
    a_image = lab_image[:, :, 1]
    b_image = lab_image[:, :, 2]
    a_image = a_image / 255.
    b_image = b_image / 255.
    # chroma calculation
    ch_image = np.sqrt(a_image**2 + b_image**2)
    return ch_image


if __name__ == '__main__':

    fn = 'results/forest.jpg'
    src = cv2.imread(fn)
    
    # Image normalization
    I = src.astype('float64') / 255.
    
    # Get chroma
    ch = Chroma(I)
    
    # Adaptive window size (for large input img)
    # cv2.namedWindow('Chroma', cv2.WINDOW_NORMAL)
    # cv2.imshow('Chroma', ch)

    cv2.imwrite('results/forest_ch.jpg', ch * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
