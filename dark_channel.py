# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys


def DarkChannel(I, w):
    M, N, _ = I.shape
    # Padded to patch_size for edge pixels
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i+w, j:j+w, :])
    return darkch


def DarkChannel_Norm(image, size):
    # normalized input
    b, g, r = cv2.split(image)
    dc = cv2.min(cv2.min(r,g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size,size))
    # Erode: only for normalized pics
    darkch = cv2.erode(dc, kernel)
    return darkch


def AtmLight(image, darkch):
    [h, w] = image.shape[:2]
    imsize = h * w
    # Number of top 0.1% pixels
    numpx = int(max(math.floor(imsize/1000),1))
    # Resize the dark channel
    darkchvec = darkch.reshape(imsize, 1)
    # Resize the original pic
    imvec = image.reshape(imsize, 3)
    # Sort dark channel
    indices = np.argsort(darkchvec, 0)
    # Get top 0.1% pixels with highest grayscale value
    indices = indices[imsize-numpx::]
    
    b, g, r = cv2.split(image)
    # Corresponded grayscale value
    gray_im = r*0.299 + g*0.587 + b*0.114
    gray_im = gray_im.reshape(imsize, 1)
    # Find the point with top 0.1% grayscale value in the original color image
    loc = np.where(gray_im == max(gray_im[indices]))
    x = loc[0][0]
    A = np.array(imvec[x])
    A = A.reshape(1, 3)
    return A


def TransmissionEstimate(image, A, size, omega):
    # default omega = 1 for haze map, 0.9 for recovered map
    im3 = np.empty(image.shape, image.dtype)

    for ind in range(3):
        # normalization with A
        im3[:, :, ind] = image[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel_Norm(im3, size)
    return transmission


def Guidedfilter(image, p, r, eps):
    # Filter the guide image and divided into several windows
    mean_I = cv2.boxFilter(image, cv2.CV_64F, (r,r))
    
    # Filter the input image
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r,r))
    mean_Ip = cv2.boxFilter(image * p, cv2.CV_64F, (r,r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(image * image, cv2.CV_64F, (r,r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Calculate the Mean matrix
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))
    
    # Linear transformation
    q = mean_a * image + mean_b
    return q


def TransmissionRefine(image, et):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 50
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t


def Recover(image, t, A, tx=0.1):
    res = np.empty(image.shape, image.dtype)
    t = cv2.max(t, tx)
    for ind in range(3):
        res[:, :, ind] = (image[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


if __name__ == '__main__':

    fn = 'results/forest.jpg'
    src = cv2.imread(fn)

    # Dark channel in [0,255]
    # darkch = DarkChannel(src, 15)
    
    # Image normalization
    I = src.astype('float64') / 255.
    
    # Patch_Size = 15*15
    # Normalized dark channel
    darkch_norm = DarkChannel_Norm(I, 15)
    # Get the atmosphere light
    A = AtmLight(I, darkch_norm)
    # Estimated transmission map
    te = TransmissionEstimate(I, A, 15, 1)
    # Refined transmission map
    t = TransmissionRefine(src, te)
    # Refined dark channel
    darkch_refined = 1 - t
    
    # For Recover: omega = 0.95
    te = TransmissionEstimate(I, A, 15, 0.95)
    t = TransmissionRefine(src, te)
    # Recovered Haze-free map
    J = Recover(I, t, A, 0.1)
    # Reset J if the grayscale value < 0
    J[J < 0] = 0
    
    # Adaptive window size (for large input img)
    # cv2.namedWindow('DarkChannel', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('DarkChannel_Refined', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Transmission', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Recovered', cv2.WINDOW_NORMAL)
    #
    # cv2.imshow('DarkChannel', darkch_norm)
    # cv2.imshow('DarkChannel_Refined', darkch_refined)
    # cv2.imshow('Transmission', t)
    # cv2.imshow('Original', src)
    # cv2.imshow('Recovered', J)

    cv2.imwrite('results/forest_darkch.jpg', darkch_refined * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)
