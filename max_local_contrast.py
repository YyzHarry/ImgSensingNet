# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

scale_local_contrast = 10

# ========================================== Variance Calculation
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
            center = np.ones((local_size, local_size, channel)) * image[i,j,:]
            var = (patch - center) ** 2
            var = var[var <= 1]
            var_image[i,j] =  np.sqrt( np.sum(var) / var.shape )
    return var_image


# ========================================== Max Local Contrast Calculation
def Max_Local_Contrast(image):
    height, width, channel = image.shape
    var_image = Local_Variance(image)
    contrast_image = np.zeros((height, width))
    # scale_local_contrast = 10, a hyperparam
    pad_size = int(scale_local_contrast/2)
    # padding default value = 0
    padimage = np.pad(var_image, ((pad_size,pad_size), (pad_size,pad_size)), 'constant')
    for i in range(height):
        for j in range(width):
            patch = padimage[i:(i+scale_local_contrast), j:(j+scale_local_contrast)]
            contrast_image[i,j] = np.max(patch)
    return contrast_image


# ========================================== Main
if __name__ == '__main__':
    # Read in by streaming? or one by one?
    fn = 'Results/8.png'

    src = cv2.imread(fn)
    # print(src)
    
    # Image normalization
    I = src.astype('float64') / 255
    
    # Patch_Size = 10*10, local_size = 5*5
    # Get local-contrast
    contrast = Max_Local_Contrast(I)
    
    # Adaptive window size (for large input img)
    cv2.namedWindow('Max_Local_Contrast', cv2.WINDOW_NORMAL)
    cv2.imshow('Max_Local_Contrast', contrast)

    # Need to rescale to [0,255]
    cv2.imwrite('Results/max_local_con.jpg', contrast * 255)
    
    cv2.waitKey(0)
