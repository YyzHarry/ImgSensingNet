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


img_count = 0 # Total training img-set
rawimgs_src_path = 'E:/ImgSensNet_videos'
dataset_save_path = 'E:/ImgSensNet_dataset' # + 'num', num in [1,6]

RESIZE_SHAPE = (128, 128, 6) # height * width * channel, channel equals to feature map nums

scale_local_contrast = 10
scale_local_sat = 10
scale_local_clr_atten = 15


rawimgs_dir = os.listdir(rawimgs_src_path)

if not os.path.exists(dataset_save_path):
        os.mkdir(dataset_save_path)

for each_rawimgs_dir in rawimgs_dir:
    print(each_rawimgs_dir)

    each_rawimgs_path = rawimgs_src_path + '/' + each_rawimgs_dir
    raw_imgs = os.listdir(each_rawimgs_path)

    for each_img in raw_imgs:
        # Get the name of each raw img data
        raw_img_name, _ = each_img.split('.')

        # Get the img full path
        img_src_full_path = each_rawimgs_path + '/' + each_img
        # Read in
        img_src_read = cv2.imread(img_src_full_path)

        # RESIZE raw imgs
        img_src_read = misc.imresize(img_src_read, [RESIZE_SHAPE[0],RESIZE_SHAPE[1]], interp='bilinear')


        #
        # ========================================== Feature 1: Refined Dark Channel
        if not os.path.exists(dataset_save_path + '/feature1'):
            os.mkdir(dataset_save_path + '/feature1')

        img_save_full_path = dataset_save_path + '/feature1' + '/'

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

        cv2.imwrite(img_save_full_path + each_img, darkch_refined * 255)

        img_count += 1
        sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 1 done." % img_count)
        sys.stdout.flush()

        #
        # ========================================== Feature 2: Max Local Contrast
        if not os.path.exists(dataset_save_path + '/feature2'):
            os.mkdir(dataset_save_path + '/feature2')

        img_save_full_path = dataset_save_path + '/feature2' + '/'

        # Image normalization
        I = img_src_read.astype('float64') / 255
        # Patch_Size = 10*10, local_size = 5*5
        # Get local-contrast
        contrast = max_local_contrast.Max_Local_Contrast(I)

        cv2.imwrite(img_save_full_path + each_img, contrast * 255)

        sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 2 done." % img_count)
        sys.stdout.flush()

        #
        # ========================================== Feature 3: Max Local Saturation
        if not os.path.exists(dataset_save_path + '/feature3'):
            os.mkdir(dataset_save_path + '/feature3')

        img_save_full_path = dataset_save_path + '/feature3' + '/'

        # Image normalization
        I = img_src_read.astype('float64') / 255
        # Patch_Size = 10*10
        # Get local-saturation
        saturation = max_local_saturation.Max_Local_Sat(I)

        cv2.imwrite(img_save_full_path + each_img, saturation * 255)

        sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 3 done." % img_count)
        sys.stdout.flush()

        #
        # ========================================== Feature 4: Min Local Color Attenuation
        if not os.path.exists(dataset_save_path + '/feature4'):
            os.mkdir(dataset_save_path + '/feature4')

        img_save_full_path = dataset_save_path + '/feature4' + '/'

        # WARNING: normalize AFTER TRANS!!! (the only one)
        # Current: After, need to preprocess in the main.py
        I = img_src_read.astype('float64')
        # Patch_Size = 15*15
        # Get local-color-attenuation
        clr_atten = min_local_clr_attenuation.Min_Local_Clr_Atten(I)

        cv2.imwrite(img_save_full_path + each_img, clr_atten * 255)

        sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 4 done." % img_count)
        sys.stdout.flush()

        #
        # ========================================== Feature 5: Hue Disparity
        if not os.path.exists(dataset_save_path + '/feature5'):
            os.mkdir(dataset_save_path + '/feature5')

        img_save_full_path = dataset_save_path + '/feature5' + '/'

        # Image normalization
        I = img_src_read.astype('float64') / 255
        # Get hue-disparity
        hue = hue_disparity.Hue_Disp(I)

        cv2.imwrite(img_save_full_path + each_img, hue * 255)

        sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 5 done." % img_count)
        sys.stdout.flush()

        #
        # ========================================== Feature 6: Chroma
        if not os.path.exists(dataset_save_path + '/feature6'):
            os.mkdir(dataset_save_path + '/feature6')

        img_save_full_path = dataset_save_path + '/feature6' + '/'

        # Image normalization
        I = img_src_read.astype('float64') / 255
        # Get chroma
        ch = chroma.Chroma(I)

        cv2.imwrite(img_save_full_path + each_img, ch * 255)

        sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 6 done." % img_count)
        sys.stdout.flush()

    print("Total processed images of %s: %d" % (each_rawimgs_dir, img_count))
    # time.sleep(600)