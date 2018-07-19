#Script for training linear model based on images for autonomous driving

import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
#to split out training and testing data
from sklearn.model_selection import train_test_split 

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

#for command line arguments
from argparse import Namespace
#for reading files
import os
import glob
import matplotlib.image as mpimg


import cv2

#taken from here:
#https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py


#settings
ROOT_PATH = "/home/sam/Desktop/CUT_Data_New/"
ROOT_FILENAME = "data1_2018_07_05_15_01_28";
MODEL_DIR = "";
#where tfrecords are
DATA_FILE = ROOT_PATH + ROOT_FILENAME + '_keras.csv'
MODEL_STORE = ROOT_PATH + MODEL_DIR #for storing the trained model for both retraining and evaluating
IMAGE_DIR = ROOT_PATH + ROOT_FILENAME + "/"

TOTAL_SAMPLES = 27078


#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(DATA_FILE)

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['image_idx']].values
    #and our steering commands as our output data
    y = data_df['steer_cmd'].values

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid =train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    #in Keras the args input handles every thing
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)) #scale 0-255 -> -1-1
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    use_restore = True
    if not os.path.exists(MODEL_STORE):
        os.makedirs(MODEL_STORE)
        use_restore = False

    #whether we should initialize weights
    if use_restore:
        model_names = glob.glob(MODEL_STORE + '*.h5')
        print("Existing Models:")
        print(model_names)
        restore_point = MODEL_STORE + 'model-006.h5'
        if os.path.isfile(restore_point):
            print ("Restoring From: ", restore_point)
            model = load_model(restore_point)

              
    #tell keras to save after each epoch
    model_var = MODEL_STORE + 'model-{epoch:03d}.h5'
    print ("Saving Models To: ",model_var)
    checkpoint = ModelCheckpoint(model_var,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    
    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator(X_train, y_train, args.batch_size, True, IMAGE_DIR),
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(X_valid, y_valid, args.batch_size, False, IMAGE_DIR),
                        validation_steps=args.validation_steps,
                        callbacks=[checkpoint],
                        verbose=1)
    
    #get random index
    rand_sample = np.random.randint(args.validation_steps)
    print ("Sample %i of %i"(rand_sample,args.validation_steps)) 


     #load up an image
    image_idx = X_valid[rand_sample][0]
    image_path = IMAGE_DIR + str(image_idx) + ".jpg"

    #load in true steering
    steering_true = y_valid[rand_sample]

    #load up the image
    cv_image = cv2.imread(image_path)
    
    print("image shape: ")
    print(cv_image.shape)

    np_image = preprocess(cv_image) # apply the preprocessing
    np_image = np.array([np_image])       # the model expects 4D array

    # predict the steering angle for the image
    steering_est = float(model.predict(np_image, batch_size=1))
        
    #print out some info
    print("Actual: ")
    print(steering_true)
    print("Prediciton: ")
    print(steering_est)
        

def model_training():
    """
    Load train/validation data set and train the model
    """
    print('Nvidia Behavioral Cloning Architecture')

    #create an args dict
    args = Namespace()
    args.test_size = 0.2
    args.keep_prob = 0.5
    args.nb_epoch = 10
    args.batch_size = 40
    args.steps_per_epoch = int( (TOTAL_SAMPLES*(1.0-args.test_size) ) / args.batch_size)
    args.validation_steps = int( (TOTAL_SAMPLES*(args.test_size) ) / args.batch_size)
    args.save_best_only = True
    args.learning_rate = 1.0e-4
    
    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    X_train, X_valid, y_train, y_valid = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5 
    train_model(model, args, X_train, X_valid, y_train, y_valid)
    
    
#function to call for predicting on an image
def model_predict():

    #what model we are using
    #model_path = model_var = MODEL_STORE + 'model-009.h5'
    model_path = MODEL_STORE + 'model-009.h5'

    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(DATA_FILE)

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['image_idx']].values
    #and our steering commands as our output data
    y = data_df['steer_cmd'].values

    #grab a random row from our samples
    num_samples = len(y)
    
    #load model (keras)
    model = load_model(model_path)


    #opencv windows
    cv2.namedWindow("Input", cv2.CV_WINDOW_AUTOSIZE)
    cv2.startWindowThread() #to make sure we can close it later on

    #images to classify
    test_samples = 100

    #setup our session
    for i in range(test_samples):

        #get random index
        rand_sample = np.random.randint(num_samples)
        print ("Sample %i of %i"%(rand_sample,num_samples) )

        #load up an image
        image_idx = X[rand_sample][0]
        image_path = IMAGE_DIR + str(image_idx) + ".jpg"

        #load in true steering
        steering_true = y[rand_sample]

        #load up the image
        cv_image = cv2.imread(image_path)


        print("image shape: ")
        print(cv_image.shape)

        np_image = preprocess(cv_image) # apply the preprocessing
        np_image = np.array([np_image])       # the model expects 4D array


        # predict the steering angle for the image
        steering_est = float(model.predict(np_image, batch_size=1))
            
        #print out some info
        print("Actual: ")
        print(steering_true)
        print("Prediciton: ")
        print(steering_est)
        
        #display image
        cv2.imshow("Input", cv_image)
        k = cv2.waitKey(0) & 0xEFFFFF
        if k == 27:
          print("Leaving...")



    #close the opencv windows
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)



    
def main():
    model_training()
    #model_predict()


if __name__ == '__main__':
    main()
