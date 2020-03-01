# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:20:27 2020

@author: ychen
"""

# =============================================================================
#  0. Load Packages
# =============================================================================

from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
from skimage import io,color
import matplotlib.pyplot as plt
from os import walk
import os
from os.path import isfile, join
import imutils
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# =============================================================================
# 1. Model
# =============================================================================

# =============================================================================
# 1.1 Define Paths
# =============================================================================
### a. For Windows
path=".images\\60-80"
model_path = path+"\\saved model\\"


### b. For Mac OS
# path="images/60-80"
# model_path=path+"/saved model/"


# =============================================================================
# 1.2 Some Functions
# =============================================================================

### a. Image Functions

#Convert RGB images to LAB using CIELAB, then return input X (L), target Y (A. B)
def rgbtolab_batch(path):
    X=[]
    Y=[]
    j=1
    row=[]
    column=[]
    image = []
    files = [f for f in os.listdir(path) if isfile(join(path, f)) and f.endswith(".jpg")==True]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=path + files[i]
        image = img_to_array(load_img(filename))
        image = np.array(image, dtype=float)
        column.append(len(image[0]))
        row.append(len(image))
        image = cv2.resize(image, (80,60))
        x = rgb2lab(1.0/255*image)[:,:,0]
        y = rgb2lab(1.0/255*image)[:,:,1:]
        y /= 128
        X.append(x)
        Y.append(y)
        j +=1
    return X,Y,row,column

#Convert LAB channels to RGB for a list of image
def labtorgb_batch(l,a,b,row,column):
    rgb=[]
    for i in range(len(l)):
        lab=np.dstack([l[i],a[i],b[i]])
        img=color.lab2rgb(lab)
        img = cv2.resize(img, (column[i],row[i]))
        rgb.append(img)
    return rgb

#Save colorized image
def savePredict(img_list,save_path):
    for i in range(len(img_list)):
        plt.imsave(("{}frame{}.jpg").format(save_path,i),img_list[i])

### b. Load Pre-trained Model and Image

#Load Model and weights
def load_model(model_name):
    json_file = open('{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("{}.h5".format(model_name))
    print("Loaded model from disk")
    return loaded_model

#Load Image
def load_image(image_path):
    X,Y,row,column = rgbtolab_batch(image_path)
    n_test=len(X)
    # Pre-process the data by reshaping
    X=np.array(X)
    Y=np.array(Y)
    X = X.reshape(n_test, 60, 80, 1)
    Y = Y.reshape(n_test, 60, 80, 2)
    return X,Y,row,column

#Load Model and Image - Colorization
def load_all(model_name,image_path):
    loaded_model=load_model(model_name)
    X,Y,row,column=load_image(image_path)
    n_test=len(X)
    loaded_model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
    loaded_model.evaluate(X,Y,batch_size=10)
    Y_predict=loaded_model.predict(X)
    Y_predict *=128
    Y_a=Y_predict[:,:,:,0].reshape(n_test,60,80,1)
    Y_b=Y_predict[:,:,:,1].reshape(n_test,60,80,1)
    Y_ori_a=Y[:,:,:,0].reshape(n_test,60,80,1)*128
    Y_ori_b=Y[:,:,:,1].reshape(n_test,60,80,1)*128
    rgb_list=labtorgb_batch(X,Y_a,Y_b,row,column)
    rgb_original=labtorgb_batch(X,Y_ori_a,Y_ori_b,row,column)
    return rgb_list,rgb_original

# =============================================================================
# 2. Colorizing Video
# =============================================================================

# =============================================================================
# 2.1. Video Paths
# =============================================================================
VIDEO="Roman Holiday Short thumnail.mp4"
bw_path = "video\\b&w_video_frames\\"
save_path = "video\\colorized_video_frames\\"
video_path = "video\\colorized_video\\"

# =============================================================================
# 2.2 Read video and convert to frames
# =============================================================================
video = "video\\"+VIDEO
width = 80
vs = cv2.VideoCapture(video)

# loop over frames from the video stream
count = 0
success = True
while success:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	success, frame = vs.read()

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if frame is None:
		break

	# resize the input frame, scale the pixel intensities to the
	frame = imutils.resize(frame, 80)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

	# resize the frame to 80x60 (the dimensions of the  model
	resized = cv2.resize(frame, (80, 60))

	# show the original frames
	cv2.imshow("Original", resized)

    # save the black and white frames
	cv2.imwrite(bw_path+"frame{}.jpg".format(str(count)), resized)
	count += 1
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

vs.release()

# close any open windows
cv2.destroyAllWindows()

# =============================================================================
#  2.3 Load model and colorize the video
# =============================================================================

# Load model on generated frames, colorize the frames and save colorized frames
colorized, original = load_all(model_path+"model_0215",bw_path)
savePredict(colorized,save_path)

# =============================================================================
#  2.4 Combine frames to a video file
# =============================================================================

### a. Define function to create a video from frames
def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and f.endswith(".jpg")==True] #If it's a valid path and exclude .DS_Store

    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

### b. Generate the video file
pathIn= save_path
pathOut = video_path+'video.avi'
fps = 30.0
convert_frames_to_video(pathIn, pathOut, fps)
print("Colorized video is saved at {}".format(pathOut))
