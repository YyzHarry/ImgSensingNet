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
train_imgs = np.empty((len(all_origin_imgs), SHAPE[0], SHAPE[1], SHAPE[2]), dtype="float32")
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
        img_src_read = misc.imresize(img_src_read, [SHAPE[0], SHAPE[1]], interp='bilinear')
        # print(img_src_read.shape)

        train_imgs[img_count, :, :, :] = img_src_read.astype('float64')/255
        # print(np.reshape(img_src_read.astype('float64')/255, (1, -1)))

        train_lbls.append( int( np.argwhere(val_lbl == int(each_org_img_aqi))[0] ) ) # prevent the same value

        img_count += 1

print("Done.")


# One-hot encoding for AQI labels
train_lbls = np_utils.to_categorical(train_lbls, num_lbl)
# train_lbls = np.reshape(np.array(train_lbls), (-1, 1))
# print(train_lbls.shape)

# Dataset splitting into train/test set
print("Splitting dataset into train & test set...")
train_imgset, test_imgset, train_aqi, test_aqi = train_test_split(train_imgs, train_lbls, test_size=0.2, random_state=0)


# ========================================== Baseline DL 2D Model
model = Sequential()

# First Conv Layer
model.add(Conv2D(4, (3, 3), padding='valid', input_shape=SHAPE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

# Second Conv Layer
model.add(Conv2D(8, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

# Third Conv Layer
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

# Flatten, DC & Output
model.add(Flatten())
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(dropout))
# model.add(Dense(1, activation='linear', name='output_layer'))
model.add(Dense(num_lbl, kernel_initializer='normal', activation='softmax', name='output_layer'))

opt = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=5,
    write_graph=True
)

model.fit(
    train_imgset, train_aqi, BATCH_SIZE, epochs=num_epoch,
    validation_data=(test_imgset, test_aqi),
    shuffle=True,
    verbose=2,
    callbacks=[
        logger
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
)

# Test error rate
test_error_rate = model.evaluate(test_imgset, test_aqi, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

# Save the model to disk
model.save("baseline_model_2d.h5")
print("Model saved to disk.")
