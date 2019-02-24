# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

SCALE_LOCAL_CONTRAST = 10


def Local_Variance(image, local_size=5):
    height, width, channel = image.shape
    var_image = np.zeros((height, width))
    pad_size = int(local_size/2)
    padimage_r = np.pad(image[:,:,0], ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=(np.inf, np.inf))
    padimage_g = np.pad(image[:,:,1], ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=(np.inf, np.inf))
    padimage_b = np.pad(image[:,:,2], ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=(np.inf, np.inf))
    padimage = np.zeros((height + pad_size*2, width + pad_size*2, 3))
    padimage[:,:,0] = padimage_r
    padimage[:,:,1] = padimage_g
    padimage[:,:,2] = padimage_b
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+local_size), j:(j+local_size), :]
            center = np.ones((local_size, local_size, channel)) * image[i, j, :]
            var = (patch - center) ** 2
            var = var[var <= 1]
            var_image[i, j] = np.sqrt(np.sum(var) / var.shape)
    return var_image


def Max_Local_Contrast(image):
    height, width, channel = image.shape
    var_image = Local_Variance(image)
    contrast_image = np.zeros((height, width))
    # default SCALE_LOCAL_CONTRAST = 10
    pad_size = int(SCALE_LOCAL_CONTRAST/2)
    # default padding val = 0
    padimage = np.pad(var_image, ((pad_size,pad_size), (pad_size,pad_size)), 'constant')
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+SCALE_LOCAL_CONTRAST), j:(j+SCALE_LOCAL_CONTRAST)]
            contrast_image[i, j] = np.max(patch)
    return contrast_image


if __name__ == '__main__':

    fn = 'results/forest.jpg'
    src = cv2.imread(fn)
    
    # Image normalization
    I = src.astype('float64') / 255.
    
    # Patch_Size = 10*10, local_size = 5*5
    # Get local-contrast
    contrast = Max_Local_Contrast(I)
    
    # Adaptive window size (for large input img)
    # cv2.namedWindow('Max_Local_Contrast', cv2.WINDOW_NORMAL)
    # cv2.imshow('Max_Local_Contrast', contrast)

    cv2.imwrite('results/forest_contrast.jpg', contrast * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
