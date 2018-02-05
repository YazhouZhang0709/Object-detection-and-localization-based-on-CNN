#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:54:38 2018

@author: cisa
"""

########   Import necessary modules and libraries   #########

from __future__ import print_function
import pandas as pd
import numpy as np
import keras
from PIL import Image
import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split
import os
import sys

#fn = sys.argv[1]
#if os.path.exists(fn):
#        print os.path.basename(fn)
##############################################################


##########   Import all the images and label   ###############
image_list = []
fn = sys.argv[-1]
if os.path.exists(fn):
    print ("File path exists")
    for filename in glob.glob(fn + '/*.jpg'):
        im = Image.open(filename)
        file_name = im.filename.split('/')[-1]
        im = np.reshape(im, (326*490*3))
        image_list.append([file_name,im])
else:
    print ("File path doesn't exist")
###############################################################

###########################    Create Pandas data frame for the data  ###########################################
image_df = pd.DataFrame(data = image_list, columns = ['name', 'data'])
image_df = image_df.set_index('name')
labels_df = pd.read_csv('find_phone_task/find_phone/labels.txt', delim_whitespace=True, names=['name', 'x', 'y'])
labels_df = labels_df.set_index('name')
df_data = image_df.join(labels_df, how='outer')
#################################################################################################################

###########################     Get the training and testing sets. Preparation for the Training      #################################
lb = [(df_data['x'].values[i]*49, df_data['y'].values[i]*32.6)for i in range(len(df_data['y'].values))] #Get the location information
X_train, X_val, y_train, y_val = train_test_split(df_data['data'].values,lb, test_size=0.07, random_state=59) # Train test split
X_train = np.stack(X_train, axis=0)
X_val = np.stack(X_val, axis=0)
y_train = np.stack(y_train, axis=0)
y_val = np.stack(y_val, axis=0)

batch_size = 10
epochs = 150
img_rows, img_cols, channels = 326, 490, 3  #Set the input size of the regression network.

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
    X_val = X_val.reshape(X_val.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)   
######################################################################################################################################

##############################        Bulid the CNN Networks by Keras    ############################################
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='valid', strides=(2, 2),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu',strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation= 'linear'))  # Here no softmax layer is used. Linear activation for regression.
model.summary()
####################################################################################################################

##########   Define the loss function and evaluation metrics for this problem   ##############
def distance_error(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)), axis=0) 

def prediction_accuracy(y_test, predictions):
    distance = y_test-predictions
    count = 0
    for dist in distance:
        dist_x = dist[0]/49
        dist_y = dist[0]/32.6
        if  np.sqrt(dist_x*dist_x+ dist_y*dist_y) < 0.05:
            count+=1
    accuracy = float(count)/(y_test.shape[0])
    return accuracy
##############################################################################################

#################################### Training Process ########################################################
model.compile(loss=[distance_error], optimizer=keras.optimizers.Adam(lr = 0.001), metrics=[distance_error])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))
##############################################################################################################

########################################  Training performance ###############################################
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
train_accuracy = prediction_accuracy(y_train, train_predictions)
val_accuracy = prediction_accuracy(y_val, val_predictions)
print('Training completed')
print('Training accuracy is ' +  str(train_accuracy))
print('Validation accuracy is ' + str(val_accuracy))
model.save('cnn_regression.h5')
print('model saved')
##############################################################################################################
