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

dataset_src_path = 'E:/ImgSensNet_videos'
origin_imgs_src_path = 'E:/ImgSensNet_dataset_mat/feature1'

SHAPE = (128, 128, 3) # height * width * channel

BATCH_SIZE = 100
num_epoch = 80
dropout = 0.2


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

# print(interval_lbl)


# ========================================== Dataset Read-in
img_count = 0 # Total origin training samples

each_locat_src_path = os.listdir(dataset_src_path)
# origin training samples w/out feature extraction
all_origin_imgs = os.listdir(origin_imgs_src_path)

# all samples, channel-last
train_imgs = np.empty((len(all_origin_imgs), SHAPE[0] * SHAPE[1] * SHAPE[2]), dtype="float32")
# train_lbls = np.empty((len(all_origin_imgs), 1), dtype="float32")
train_lbls = []


for each_locat_path in each_locat_src_path:
    each_rawimgs_path = dataset_src_path + '/' + each_locat_path
    raw_imgs = os.listdir(each_rawimgs_path)

    for each_img in raw_imgs:
        sys.stdout.write("\rReading all training data into memory... %d/%d " % (img_count + 1, len(all_origin_imgs)))
        sys.stdout.flush()

        # Get the name of each raw img data
        raw_img_name, _ = each_img.split('.')
        _, _, each_org_img_aqi, _ = raw_img_name.split('_')

        # Get the img full path
        img_src_full_path = each_rawimgs_path + '/' + each_img
        # Read in
        img_src_read = cv2.imread(img_src_full_path)

        # RESIZE raw imgs
        img_src_read = misc.imresize(img_src_read, [SHAPE[0],SHAPE[1]], interp='bilinear')
        # print(img_src_read.shape)

        train_imgs[img_count, :] = np.reshape(img_src_read.astype('float64')/255, (1, -1))
        # print(np.reshape(img_src_read.astype('float64')/255, (1, -1)))

        train_lbls.append( int( np.argwhere(val_lbl == int(each_org_img_aqi))[0] ) ) # prevent the same value

        img_count += 1

print("Done.")


# One-hot encoding for AQI labels
# train_lbls = np_utils.to_categorical(train_lbls, num_lbl)
train_lbls = np.reshape(np.array(train_lbls), (-1, 1))
# print(train_lbls.shape)
# print(train_lbls.sum())

# Dataset splitting into train/test set
print("Splitting dataset into train & test set...")
train_imgset, test_imgset, train_aqi, test_aqi = train_test_split(train_imgs, train_lbls, test_size=0.2, random_state=0)


# ========================================== Baseline Model
# kNN
neigh_knn = KNeighborsClassifier(n_neighbors=5)
# DNN
neigh_dnn = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                          learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                          random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                          early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# SVM
neigh_svm = svm.SVC()
# MLR
neigh_mlr = linear_model.SGDClassifier()
# DTree
neigh_dtree = DecisionTreeClassifier(random_state=0)
# RFC
neigh_rfc = RandomForestClassifier(max_depth=2, random_state=0)

# Train
neigh_knn.fit(train_imgset, train_aqi)
neigh_dnn.fit(train_imgset, train_aqi)
neigh_svm.fit(train_imgset, train_aqi)
neigh_mlr.fit(train_imgset, train_aqi)
neigh_dtree.fit(train_imgset, train_aqi)
neigh_rfc.fit(train_imgset, train_aqi)

# Test Accuracy
neigh_knn_cnt = 0
neigh_dnn_cnt = 0
neigh_svm_cnt = 0
neigh_mlr_cnt = 0
neigh_dtree_cnt = 0
neigh_rfc_cnt = 0

for i in range(test_imgset.shape[0]):
    # kNN
    if neigh_knn.predict(test_imgset[i].reshape(1, -1)) == test_aqi[i]:
        neigh_knn_cnt += 1
    # DNN
    if neigh_dnn.predict(test_imgset[i].reshape(1, -1)) == test_aqi[i]:
        neigh_dnn_cnt += 1
    # SVM
    if neigh_svm.predict(test_imgset[i].reshape(1, -1)) == test_aqi[i]:
        neigh_svm_cnt += 1
    # MLR
    if neigh_mlr.predict(test_imgset[i].reshape(1, -1)) == test_aqi[i]:
        neigh_mlr_cnt += 1
    # Dtree
    if neigh_dtree.predict(test_imgset[i].reshape(1, -1)) == test_aqi[i]:
        neigh_dtree_cnt += 1
    # RFC
    if neigh_rfc.predict(test_imgset[i].reshape(1, -1)) == test_aqi[i]:
        neigh_rfc_cnt += 1

print("Accuracy of baseline kNN is: %.4f\n" % (neigh_knn_cnt/test_imgset.shape[0]))
print("Accuracy of baseline DNN is: %.4f\n" % (neigh_dnn_cnt/test_imgset.shape[0]))
print("Accuracy of baseline MLR is: %.4f\n" % (neigh_mlr_cnt/test_imgset.shape[0]))
print("Accuracy of baseline SVM is: %.4f\n" % (neigh_svm_cnt/test_imgset.shape[0]))
print("Accuracy of baseline DTree is: %.4f\n" % (neigh_dtree_cnt/test_imgset.shape[0]))
print("Accuracy of baseline RFC is: %.4f\n" % (neigh_rfc_cnt/test_imgset.shape[0]))

with open('result_baseline_classical_ml.txt', 'w') as f:
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


# Save Model
with open('neigh_knn.pickle', 'wb') as f:
    pickle.dump(neigh_knn, f)

with open('neigh_dnn.pickle', 'wb') as f:
    pickle.dump(neigh_dnn, f)

with open('neigh_svm.pickle', 'wb') as f:
    pickle.dump(neigh_svm, f)

with open('neigh_mlr.pickle', 'wb') as f:
    pickle.dump(neigh_mlr, f)

with open('neigh_dtree.pickle', 'wb') as f:
    pickle.dump(neigh_dtree, f)

with open('neigh_rfc.pickle', 'wb') as f:
    pickle.dump(neigh_rfc, f)


'''
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))
'''