#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:09:11 2018

@author: cisa
"""

############   Import necessary modules   ##########
import os
import sys
from PIL import Image
import numpy as np
from keras.models import load_model
from keras import backend as K
####################################################

##########################  Load previous trained model   ################################
def distance_error(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)), axis=0) 
model = load_model('cnn_regression.h5', custom_objects={'distance_error': distance_error})
##########################################################################################

###############     Load test image   ####################
file_path = sys.argv[-1]
if os.path.exists(file_path):
    print ("File path exists.")
    im = Image.open(file_path)
    file_name = im.filename.split('/')[-1]
    print(type(im))
    im = np.reshape(im, (326*490*3))
else:
    print ("File path doesn't exist!")
##########################################################   
   

###################      Reshape the image for training     ############################## 
img_rows, img_cols, channels = 326, 490, 3  #Set the input size of the regression network.
if K.image_data_format() == 'channels_first':
    im= im.reshape(1, channels, img_rows, img_cols)
    input_shape = (1, channels, img_rows, img_cols)
else:
    im = im.reshape(1, img_rows, img_cols, channels)
    input_shape = (1, img_rows, img_cols, channels)
##########################################################################################    


######################   Show the prediction result   ############################
cal_result = model.predict(im)
predict_result = [round(cal_result[0][0]/49.0, 4), round(cal_result[0][1]/32.6,4)] 
print(predict_result)
##################################################################################