import cv2

import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import csv

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


#Read in video and csv data file and convert to full path per the format used here:
#https://github.com/llSourcell/How_to_simulate_a_self_driving_car


# Video file to splice
ROOT_PATH = "/home/sam/Desktop/CUT_Data_New/"
ROOT_FILENAME = "data1_2018_07_17_21_07_10";
PATH_TO_VIDEO = ROOT_PATH + ROOT_FILENAME + '.avi'
PATH_TO_CSV = ROOT_PATH + ROOT_FILENAME + '.csv'
OUT_CSV = ROOT_PATH + ROOT_FILENAME + '_keras.csv'
OUT_PATH = ROOT_PATH + ROOT_FILENAME + '/'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

print("Opening CSV File Writer")
csvfile = open(OUT_CSV, 'w')
label_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

full_row = []
full_row.extend(['image_idx'])
full_row.extend(['time'])
full_row.extend(['propel_cmd'])
full_row.extend(['steer_cmd'])
label_writer.writerow(full_row) 

print("Opening CSV File")
#columns=['image_idx','time','propel_cmd','steer_cmd','height_cmd','tilt_cmd','propel_est','steer_est']
full_data = pd.read_csv(PATH_TO_CSV)
np_data = full_data.values

#extract columns
image_idx_np = full_data[['image_idx']].values.flatten()
time_np = full_data[['time']].values.flatten()
propel_np = full_data[['propel_cmd']].values.flatten()
steer_np = full_data[['steer_cmd']].values.flatten()

#filename_np = full_data[['filename']].values.flatten()
#print(time_np)

#exit

#get csv info
rowsize = len(full_data.columns)
print("Number of Columns: ", rowsize)
num_frames_csv = int(image_idx_np[-1]) #-1 gives last element
print("Number of Frames CSV: ", num_frames_csv )
row_idx = 0

print("Opening Video File")
cv2.namedWindow("VideoCapture", 1)
capture = cv2.VideoCapture(PATH_TO_VIDEO)
cv2.startWindowThread() #to make sure we can close it later on

print("Running Image Parsing")
 
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
        print("No More Rows In CSV:", row_idx)
        break

    #grab the current row info from csv file and compare with frame idx
    propel_cmd = propel_np[row_idx]
    steer_cmd =  steer_np[row_idx]
    image_idx = image_idx_np[row_idx]
    time = time_np[row_idx]#second to last
#    temp_filename = str(csv_idx)
#    print(temp_filename)
#  
    if(image_idx != row_idx):
        print("Frame Indices do not match! ", row_idx, image_idx)
        break
    row_idx = row_idx + 1

    #save frame to file with csv_idx as filename
    img_filename = OUT_PATH +str(image_idx) + ".jpg"
    #img_filename = 'test_output.txt'

    cv2.imwrite(img_filename,image_np)

    #write to new csv
    full_row = []
    full_row.extend([image_idx])
    full_row.extend([time])
    #full_row.extend([img_filename])
    full_row.extend([propel_cmd])
    full_row.extend([steer_cmd])
    label_writer.writerow(full_row)  

    #display frame
    cv2.imshow("VideoCapture", image_np)
    k = cv2.waitKey(5) & 0xEFFFFF
    if k == 27:
        print("You Pressed Escape")
        break
            
print("Video Parsing Is Finished")
print("Images Saved To: ", OUT_PATH)
print("Total Samples", image_idx)
print("Closing CSV")
csvfile.close()
print("Closing Video File")
capture.release()
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)
