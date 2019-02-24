# -*- coding: utf-8 -*-

import cv2
import os, sys
import time
from scipy import misc
import argparse

import chroma
import dark_channel
import hue_disparity
import max_local_contrast
import max_local_saturation
import min_local_clr_attenuation

RESIZE_SHAPE = (128, 128, 6)


def extract_feature(args):
    img_count = 0
    rawimgs_src_path = args.data_path
    dataset_save_path = args.save_path

    rawimgs_dir = os.listdir(rawimgs_src_path)

    if not os.path.exists(dataset_save_path):
            os.mkdir(dataset_save_path)

    for each_rawimgs_dir in rawimgs_dir:
        print(each_rawimgs_dir)

        each_rawimgs_path = rawimgs_src_path + '/' + each_rawimgs_dir
        raw_imgs = os.listdir(each_rawimgs_path)

        for each_img in raw_imgs:
            raw_img_name, _ = each_img.split('.')
            img_src_full_path = each_rawimgs_path + '/' + each_img
            img_src_read = cv2.imread(img_src_full_path)
            # resize raw imgs to 128*128
            img_src_read = misc.imresize(img_src_read, [RESIZE_SHAPE[0], RESIZE_SHAPE[1]], interp='bilinear')

            # ====================== Feature 1: Refined Dark Channel
            if not os.path.exists(dataset_save_path + '/feature1'):
                os.mkdir(dataset_save_path + '/feature1')

            img_save_full_path = dataset_save_path + '/feature1/'

            I = img_src_read.astype('float64') / 255.
            # Patch_Size = 15*15
            darkch_norm = dark_channel.DarkChannel_Norm(I, 15)
            A = dark_channel.AtmLight(I, darkch_norm)
            te = dark_channel.TransmissionEstimate(I, A, 15, 1)
            t = dark_channel.TransmissionRefine(img_src_read, te)
            darkch_refined = 1 - t

            cv2.imwrite(img_save_full_path + each_img, darkch_refined * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            img_count += 1
            sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 1 done." % img_count)
            sys.stdout.flush()

            # ====================== Feature 2: Max Local Contrast
            if not os.path.exists(dataset_save_path + '/feature2'):
                os.mkdir(dataset_save_path + '/feature2')

            img_save_full_path = dataset_save_path + '/feature2/'

            I = img_src_read.astype('float64') / 255.
            # Patch_Size = 10*10, local_size = 5*5
            contrast = max_local_contrast.Max_Local_Contrast(I)

            cv2.imwrite(img_save_full_path + each_img, contrast * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 2 done." % img_count)
            sys.stdout.flush()

            # ====================== Feature 3: Max Local Saturation
            if not os.path.exists(dataset_save_path + '/feature3'):
                os.mkdir(dataset_save_path + '/feature3')

            img_save_full_path = dataset_save_path + '/feature3/'

            I = img_src_read.astype('float64') / 255.
            # Patch_Size = 10*10
            saturation = max_local_saturation.Max_Local_Sat(I)

            cv2.imwrite(img_save_full_path + each_img, saturation * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 3 done." % img_count)
            sys.stdout.flush()

            # ====================== Feature 4: Min Local Color Attenuation
            if not os.path.exists(dataset_save_path + '/feature4'):
                os.mkdir(dataset_save_path + '/feature4')

            img_save_full_path = dataset_save_path + '/feature4/'

            # normalize AFTER TRANS
            I = img_src_read.astype('float64')
            # Patch_Size = 15*15
            clr_atten = min_local_clr_attenuation.Min_Local_Clr_Atten(I)

            cv2.imwrite(img_save_full_path + each_img, clr_atten * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 4 done." % img_count)
            sys.stdout.flush()

            # ====================== Feature 5: Hue Disparity
            if not os.path.exists(dataset_save_path + '/feature5'):
                os.mkdir(dataset_save_path + '/feature5')

            img_save_full_path = dataset_save_path + '/feature5/'

            I = img_src_read.astype('float64') / 255.
            hue = hue_disparity.Hue_Disp(I)

            cv2.imwrite(img_save_full_path + each_img, hue * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 5 done." % img_count)
            sys.stdout.flush()

            # ====================== Feature 6: Chroma
            if not os.path.exists(dataset_save_path + '/feature6'):
                os.mkdir(dataset_save_path + '/feature6')

            img_save_full_path = dataset_save_path + '/feature6/'

            I = img_src_read.astype('float64') / 255.
            ch = chroma.Chroma(I)

            cv2.imwrite(img_save_full_path + each_img, ch * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            sys.stdout.write("\rProcessing a new image: %d ...... Feature maps extraction: 6 done." % img_count)
            sys.stdout.flush()

        print("Total processed images of %s: %d" % (each_rawimgs_dir, img_count))
        # time.sleep(600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True,
                        help='directory containing raw images')
    parser.add_argument('--save-path', type=str, required=True,
                        help='directory for saving processed feature images')
    # parser.add_argument('--contrast-scale', type=int, default=10,
    #                     help='scale factor for maximum local contrast feature')
    # parser.add_argument('--sat-scale', type=int, default=10,
    #                     help='scale factor for maximum local saturation feature')
    # parser.add_argument('--clratten-scale', type=int, default=15,
    #                     help='scale factor for minimum local color attenuation feature')

    args = parser.parse_args()
    extract_feature(args)


if __name__ == '__main__':
    main()
