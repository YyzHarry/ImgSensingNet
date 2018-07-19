# -*- coding: utf-8 -*-

import cv2
import os, sys
import time
from scipy import misc

import chroma
import dark_channel
import hue_disparity
import max_local_contrast
import max_local_saturation
import min_local_clr_attenuation

scale_local_contrast = 10
scale_local_sat = 10
scale_local_clr_atten = 15

target_img_name = 'forest'

img_src_read = cv2.imread('results/' + target_img_name + '.jpg')

# RESIZE raw imgs
# img_src_read = misc.imresize(img_src_read, [RESIZE_SHAPE[0],RESIZE_SHAPE[1]], interp='bilinear')

#
# ========================================== Feature 1: Refined Dark Channel
# Image normalization
I = img_src_read.astype('float64') / 255
# Patch_Size = 15*15
# Normalized dark channel
darkch_norm = dark_channel.DarkChannel_Norm(I, 15)
# Get the atmosphere light
A = dark_channel.AtmLight(I, darkch_norm)
# Estimated transmisstion map
te = dark_channel.TransmissionEstimate(I, A, 15, 1)
# Refined transmisstion map
t = dark_channel.TransmissionRefine(img_src_read, te)
# Refined dark channel
darkch_refined = 1 - t

#
# ========================================== Feature 2: Max Local Contrast
contrast = max_local_contrast.Max_Local_Contrast(I)

#
# ========================================== Feature 3: Max Local Saturation
saturation = max_local_saturation.Max_Local_Sat(I)

#
# ========================================== Feature 4: Min Local Color Attenuation
# WARNING: normalize AFTER TRANS!!! (the only one)
# Current: After, need to preprocess in the main.py
I2 = img_src_read.astype('float64')
# Patch_Size = 15*15
# Get local-color-attenuation
clr_atten = min_local_clr_attenuation.Min_Local_Clr_Atten(I2)

#
# ========================================== Feature 5: Hue Disparity
hue = hue_disparity.Hue_Disp(I)

#
# ========================================== Feature 6: Chroma
ch = chroma.Chroma(I)


# Need to rescale to [0,255]
cv2.imwrite('results/' + target_img_name + '_darkch.jpg', darkch_refined * 255)
cv2.imwrite('results/' + target_img_name + '_contrast.jpg', contrast * 255)
cv2.imwrite('results/' + target_img_name + '_saturat.jpg', saturation * 255)
cv2.imwrite('results/' + target_img_name + '_clratten.jpg', clr_atten * 255)
cv2.imwrite('results/' + target_img_name + '_hue.jpg', hue * 255)
cv2.imwrite('results/' + target_img_name + '_ch.jpg', ch * 255)