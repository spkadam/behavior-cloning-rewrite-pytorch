import cv2

import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential, load_model
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#helper class to define input shape and generate training images given image paths & steering angles
from utils_CUT import INPUT_SHAPE, batch_generator, preprocess
#from video_2_keras_csv import image_idx
#for command line arguments
from argparse import Namespace

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

PREDICT = True
model_path = "/home/sam/Desktop/CUT_Data_New/model-009.h5"

# Video file to splice
ROOT_PATH = "/home/sam/Desktop/CUT_Data_New/"
#ROOT_PATH = "/home/lt44243/tensorflow_data/CUT_Data/"
ROOT_FILENAME = "data1_2018_07_17_21_07_10";
PATH_TO_VIDEO = ROOT_PATH + ROOT_FILENAME + '.avi'
PATH_TO_CSV = ROOT_PATH + ROOT_FILENAME + '.csv'

print "Opening CSV File"
full_data = pd.read_csv(PATH_TO_CSV)
np_data = full_data.values
rowsize = len(np_data[0])
print "Number of Columns: ", rowsize
num_frames_csv = int(np_data[-1,0]) #-1 gives last element
print "Number of Frames CSV: ", num_frames_csv 
row_idx = 0

print "Opening Video File"
cv2.namedWindow("Steering", 1)
capture = cv2.VideoCapture(PATH_TO_VIDEO)
cv2.startWindowThread() #to make sure we can close it later on

if(PREDICT):
  #load model (keras)
  model = load_model(model_path)
 
while True:
  ret, image_np = capture.read()

  try:
    if not image_np.size:
        print("Video Capture Failed")
        break
  except:
    break

  #grab current row from csv file
  try:
    row_data = np_data[row_idx]
  except:
    print "No More Rows In CSV:", row_idx
    break
  #grab the current index from csv file and compare with frame idx
  csv_idx = int(row_data[0]) #first element
  if(csv_idx != row_idx):
    print "Frame Indices do not match! ", row_idx, csv_idx
    break
  row_idx = row_idx + 1

  if(PREDICT):
    np_image = preprocess(image_np) # apply the preprocessing
    np_image = np.array([np_image])       # the model expects 4D array
    # predict the steering angle for the image
    steer_est = -float(model.predict(np_image, batch_size=1))

  #grab the steer value
  steer_val = row_data[3]
  height, width, channels = image_np.shape
  steer_disp = int( (float(width)/2.0) + (steer_val * float(width)/2.0) ) 
  topline = 10

  if(PREDICT):
    steer_disp_pred = int( (float(width)/2.0) + (steer_est * float(width)/2.0) )
    cv2.circle(image_np, (steer_disp_pred, topline), 10, (255,0,0), -1)

  #display steer on image
  cv2.circle(image_np, (steer_disp, topline), 10, (0,0,255), -1)
  cv2.line(image_np, (int(float(width)/2.0),0),(int(float(width)/2.0),25),(0,0,255),1)


  #grab the time value and write on image
  timeval = row_data[1]
  timestr = str(timeval)
  cv2.putText(image_np, timestr, (10,height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

  
  
  #display frame
  cv2.imshow("VideoCapture", image_np)
  k = cv2.waitKey(30) & 0xEFFFFF
  if k == 27:
      print("You Pressed Escape")
      break
            
print("Exiting...")

capture.release()
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)
