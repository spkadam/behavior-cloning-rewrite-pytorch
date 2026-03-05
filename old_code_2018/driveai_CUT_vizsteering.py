import cv2
import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# Video file to splice
ROOT_PATH = "/home/sam/Desktop/CUT_Data_New/"
#ROOT_PATH = "/home/lt44243/tensorflow_data/CUT_Data/"
ROOT_FILENAME = "data1_2018_07_17_21_07_10";
PATH_TO_VIDEO = ROOT_PATH + ROOT_FILENAME + '.avi'
PATH_TO_CSV = ROOT_PATH + ROOT_FILENAME + '.csv'

print("Opening CSV File")
full_data = pd.read_csv(PATH_TO_CSV)
np_data = full_data.values
rowsize = len(np_data[0])
print ("Number of Columns: ", rowsize)
num_frames_csv = int(np_data[-1,0]) #-1 gives last element
print ("Number of Frames CSV: ", num_frames_csv )
row_idx = 0

print ("Opening Video File")
window = cv2.namedWindow("Steering",1)
capture = cv2.VideoCapture(PATH_TO_VIDEO)
cv2.startWindowThread() #to make sure we can close it later on
 
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
    print ("No More Rows In CSV:", row_idx)
    break
  #grab the current index from csv file and compare with frame idx
  csv_idx = int(row_data[0]) #first element
  if(csv_idx != row_idx):
    print ("Frame Indices do not match! ", row_idx, csv_idx)
    break
  row_idx = row_idx + 1

  #grab the steer value
  steer_val = row_data[3]
  height, width, channels = image_np.shape
  steer_disp = int( (float(width)/2.0) + (steer_val * float(width)/2.0) ) 
  topline = 10

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
