# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import os, sys
import time
import pickle
from scipy import misc

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

dataset_src_path = 'E:/ImgSensNet_videos_all'
origin_imgs_src_path = 'E:/ImgSensNet_dataset_mat/feature1'

SHAPE = (128, 128, 3) # height * width * channel


# ========================================== Label AQI Intervals & Ouput Defination
print("Labeling dataset...")
lbls = os.listdir(dataset_src_path)
num_lbl = len(lbls) # Total label numbers (AQI Intervals)
val_lbl = np.empty([num_lbl], np.int32)
interval_lbl = np.zeros((num_lbl+1))
# Define max AQI possible value
interval_lbl[-1] = 500
num_lbl = 0

for each_lbl in lbls:
    _, _, aqi = each_lbl.split('_')
    val_lbl[num_lbl] = int(aqi)
    num_lbl += 1

val_lbl.sort()
# print(val_lbl)

for i in range(num_lbl-1):
    interval_lbl[i+1] = (val_lbl[i] + val_lbl[i+1]) / 2.0


# ========================================== Load Baselines
print("Loading baseline models...")
with open('neigh_knn.pickle', 'rb') as f:
    neigh_knn = pickle.load(f)

with open('neigh_dnn.pickle', 'rb') as f:
    neigh_dnn = pickle.load(f)

with open('neigh_svm.pickle', 'rb') as f:
    neigh_svm = pickle.load(f)

with open('neigh_mlr.pickle', 'rb') as f:
    neigh_mlr = pickle.load(f)

with open('neigh_dtree.pickle', 'rb') as f:
    neigh_dtree = pickle.load(f)

with open('neigh_rfc.pickle', 'rb') as f:
    neigh_rfc = pickle.load(f)


# ========================================== All Dataset Read-in
img_count = 0 # Total origin training samples

# MSE for baselines
MSE_neigh_knn = 0
MSE_neigh_dnn = 0
MSE_neigh_svm = 0
MSE_neigh_mlr = 0
MSE_neigh_dtree = 0
MSE_neigh_rfc = 0

each_locat_src_path = os.listdir(dataset_src_path)

for each_locat_path in each_locat_src_path:
    each_rawimgs_path = dataset_src_path + '/' + each_locat_path
    raw_imgs = os.listdir(each_rawimgs_path)

    for each_img in raw_imgs:
        sys.stdout.write("\rPerforming evaluations on all img data... %d " % (img_count + 1))
        sys.stdout.flush()

        # Get the name of each raw img data
        raw_img_name, _ = each_img.split('.')
        _, _, each_org_img_aqi, _ = raw_img_name.split('_')

        # Get the img full path
        img_src_full_path = each_rawimgs_path + '/' + each_img
        # Read in
        img_src_read = cv2.imread(img_src_full_path)

        # RESIZE raw imgs
        img_src_read = misc.imresize(img_src_read, [SHAPE[0], SHAPE[1]], interp='bilinear')
        # print(img_src_read.shape)

        # Predicted val v.s. Ground truth
        # kNN
        MSE_neigh_knn += math.pow(neigh_knn.predict(np.reshape(img_src_read.astype('float64')/255, (1, -1))) -\
                                  int(np.argwhere(val_lbl == int(each_org_img_aqi))[0]), 2)
        # DNN
        MSE_neigh_dnn += math.pow(neigh_dnn.predict(np.reshape(img_src_read.astype('float64') / 255, (1, -1))) -\
                                  int(np.argwhere(val_lbl == int(each_org_img_aqi))[0]), 2)
        # SVM
        MSE_neigh_svm += math.pow(neigh_svm.predict(np.reshape(img_src_read.astype('float64') / 255, (1, -1))) -\
                                  int(np.argwhere(val_lbl == int(each_org_img_aqi))[0]), 2)
        # MLR
        MSE_neigh_mlr += math.pow(neigh_mlr.predict(np.reshape(img_src_read.astype('float64') / 255, (1, -1))) -\
                                  int(np.argwhere(val_lbl == int(each_org_img_aqi))[0]), 2)
        # DTree
        MSE_neigh_dtree += math.pow(neigh_dtree.predict(np.reshape(img_src_read.astype('float64') / 255, (1, -1))) -\
                                    int(np.argwhere(val_lbl == int(each_org_img_aqi))[0]), 2)
        # RFC
        MSE_neigh_rfc += math.pow(neigh_rfc.predict(np.reshape(img_src_read.astype('float64') / 255, (1, -1))) -\
                                  int(np.argwhere(val_lbl == int(each_org_img_aqi))[0]), 2)

        img_count += 1

print("Done.")

print("MSE, RMSE of baseline kNN is: %.4f, %.4f\n" % (MSE_neigh_knn/img_count, math.sqrt(MSE_neigh_knn/img_count)))
print("MSE, RMSE of baseline DNN is: %.4f, %.4f\n" % (MSE_neigh_dnn/img_count, math.sqrt(MSE_neigh_dnn/img_count)))
print("MSE, RMSE of baseline MLR is: %.4f, %.4f\n" % (MSE_neigh_mlr/img_count, math.sqrt(MSE_neigh_mlr/img_count)))
print("MSE, RMSE of baseline SVM is: %.4f, %.4f\n" % (MSE_neigh_svm/img_count, math.sqrt(MSE_neigh_svm/img_count)))
print("MSE, RMSE of baseline DTree is: %.4f, %.4f\n" % (MSE_neigh_dtree/img_count, math.sqrt(MSE_neigh_dtree/img_count)))
print("MSE, RMSE of baseline RFC is: %.4f, %.4f\n" % (MSE_neigh_rfc/img_count, math.sqrt(MSE_neigh_rfc/img_count)))

'''
with open('rmse_baseline_classical.txt', 'w') as f:
    f.write("k-Nearest Neighbors: ")
    f.write(str(neigh_knn_cnt/test_imgset.shape[0]))
    f.write('\n')
    f.write("Deep Neural Networks: ")
    f.write(str(neigh_dnn_cnt / test_imgset.shape[0]))
    f.write('\n')
    f.write("Support Vector Machines: ")
    f.write(str(neigh_svm_cnt / test_imgset.shape[0]))
    f.write('\n')
    f.write("Multi-var Linear Classification: ")
    f.write(str(neigh_mlr_cnt / test_imgset.shape[0]))
    f.write('\n')
    f.write("Decision Tree Classification: ")
    f.write(str(neigh_dtree_cnt / test_imgset.shape[0]))
    f.write('\n')
    f.write("Random Forest Classification: ")
    f.write(str(neigh_rfc_cnt / test_imgset.shape[0]))
    f.write('\n')
'''
