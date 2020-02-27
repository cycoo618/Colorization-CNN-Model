# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:40:28 2020

@author: ychen
"""
# =============================================================================
#  0. Load Packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage import io,color
from os import walk
import os
import random
from datetime import datetime
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab


###For Windows

path="images\\60-80"
test_path=path+"\\test\\"
predict_path=path+"\\predict\\"
model_path = path+"\\saved model\\"

###For Mac OS
# =============================================================================
# path=".images/60-80"
# test_path=path+"/test/"
# predict_path=path+"/predict/"
# model_path = path+"/saved model/"
# =============================================================================

def rgbtolab_batch(path):
    X=[]
    Y=[]
    j=1

    for (dirpath,dirnames,filenames) in walk(path):
        for filename in filenames:
            if filename.endswith(".jpg")==True:
                # Get images
                image = img_to_array(load_img(dirpath+filename))
                image = np.array(image, dtype=float)
                column = len(image[0])
                row = len(image)
                x = rgb2lab(1.0/255*image)[:,:,0]
                y = rgb2lab(1.0/255*image)[:,:,1:]
                y /= 128

                X.append(x)
                Y.append(y)
                j +=1
                n_test=len(X)

    return X,Y,row,column

def labtorgb_batch(l,a,b):
    rgb=[]
    for i in range(len(l)):
        lab=np.dstack([l[i],a[i],b[i]])
        img=color.lab2rgb(lab)
        rgb.append(img)

    return rgb

def showgray(path):
    for (dirpath,dirnames,filenames) in walk(path):
        for filename in filenames:
            rgb=io.imread(dirpath+filename,plugin='matplotlib')
            lab=color.rgb2lab(rgb)
            lab_copy=lab.copy()
            lab_copy[:,:,1]=0
            lab_copy[:,:,2]=0
            grey=color.lab2rgb(lab_copy)
            #plt.imsave(img[:-4]+'_gray.jpg',grey)
            plt.imshow(grey)
            plt.show()

def showRGB(img_list):
    for i in img_list:
        plt.imshow(i)
        plt.show()

def savePredict(img_list):
    for i in range(len(img_list)):
        plt.imsave(("{}{}_predict{}.jpg").format(predict_path,i),datetime.today().strftime('%y%m%d-%h%m%s'),img_list[i])

##Load Model and weights
def load_model(model_name):

    json_file = open('{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("{}.h5".format(model_name))
    print("Loaded model from disk")

    return loaded_model

def load_image(image_path):
    X,Y,row,column = rgbtolab_batch(image_path)
    n_test=len(X)
    # Pre-process the data by reshaping
    X=np.array(X)
    Y=np.array(Y)
    X = X.reshape(n_test, row, column, 1)
    Y = Y.reshape(n_test, row, column, 2)

    return X,Y,row,column

def load_all(model_name,image_path):
    loaded_model=load_model(model_name)
    X,Y,row,column=load_image(image_path)
    n_test=len(X)
    loaded_model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
    loaded_model.evaluate(X,Y,batch_size=10)
    Y_predict=loaded_model.predict(X)
    Y_predict *=128
    Y_a=Y_predict[:,:,:,0].reshape(n_test,row,column,1)
    Y_b=Y_predict[:,:,:,1].reshape(n_test,row,column,1)
    Y_ori_a=Y[:,:,:,0].reshape(n_test,row,column,1)*128
    Y_ori_b=Y[:,:,:,1].reshape(n_test,row,column,1)*128
    rgb_list=labtorgb_batch(X,Y_a,Y_b)
    rgb_original=labtorgb_batch(X,Y_ori_a,Y_ori_b)

    return rgb_list,rgb_original

# Load model and image path, get colorized images and original images
colorized, original = load_all(model_path+"model_0215",test_path)

# print colorized images and original images, save colorized images
print("Colorized Images:")
showRGB(colorized)
print("Original Images")
showRGB(original)
savePredict(colorized)
print("Predicted images saved to {}".format(predict_path))
