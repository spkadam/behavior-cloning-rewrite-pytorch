import cv2
import numpy as np

#Modifed from
#https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/utils.py


#IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 150, 320, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
OUTPUT_SHAPE = 1 #just steer for now

def load_image(image_file):
    """
    Load RGB images from a complete filepath (image_file)
    Don't use matplot image
    """
    result = cv2.imread(image_file)
    return result

def resize(image):
    """
    Resize image
    """
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return image 

def crop(image):
    """
    Crop the image Disabled)
    """
    image = image[180:480, 0:640, :] # focuses on road
    return image 


def rgb2hsv(image):
    """
    Convert the image from BGR (opencv) to hsv
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2hsv(image)
    return image


def image_brightness(image, steering_angle):
    """
    Randomly adjust image brightness
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image, steering_angle

def image_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def image_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.006 #uncalibrated
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def augument(image_path, steering_angle, range_x=30, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image = load_image(image_path)
    image, steering_angle = image_flip(image, steering_angle)
    image, steering_angle = image_translate(image, steering_angle, range_x, range_y)
    image, steering_angle = image_brightness(image, steering_angle)
    return image, steering_angle


def batch_generator(image_paths, steering_angles, batch_size, is_training, image_dir):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        #loop through images and fill up batch size
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            image_idx = image_paths[index][0]
            image_file = image_dir + str(image_idx) + ".jpg"
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(image_file, steering_angle)
            else:
                image = load_image(image_file) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, (-(steers))


def TestFunctions():
    #180,18,1,-0.157,0,0,1.242,0.0083565
    #For testing the functions
    image_path = '/home/ubuntu/Desktop/CUT_Data_New/data1_2018_07_05_15_01_28/272.jpg'
    steer = -0.296

    cv2.namedWindow("RawImage", 1)
    cv2.namedWindow("Preprocessed", 1)
    cv2.startWindowThread() #to make sure we can close it later on

    #load
    image = load_image( image_path )
    print("Raw Shape")
    print(image.shape)

    #Test augment
    image, steer = image_brightness(image, steer)
    image, steer = image_flip(image, steer)
    image, steer = image_translate(image, steer, range_x=30, range_y=10)

    print(float(-(steer)))

    #display image with line
    cv2.imshow("RawImage", image)
    cv2.waitKey(0)

    #display preprocess
    process_image = preprocess(image)
    cv2.imshow("Preprocessed", process_image)
    cv2.waitKey(0)

    print(process_image.shape)

    # hue
    hmin = image[..., 0].min()
    hmax = image[..., 0].max()
    # sat
    smin = image[..., 1].min()
    smax = image[..., 1].max()
    # value
    vmin = image[..., 2].min()
    vmax = image[..., 2].max()

    print ("Limits:",hmin,hmax,smin,smax,vmin,vmax)

    #close
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)


#TestFunctions()
