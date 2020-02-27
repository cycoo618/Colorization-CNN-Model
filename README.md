# Colorization-CNN-Model  

## Use CNN to colorize black and white images/videos

For each of the .py file, there is equivalent jupyter notebook file, which is an easy and quick way to explore.

### I. Training:  
run file: Colorization_training.py  

##### Process logic:  

#### 1. Data Process: CIELAB  
  1.1 Download images from [Unsample.net](Unsample.net)  

  1.2 Use CIELAB method to convert RGB data to LAB data.  

      - L: lightness  
      - A: from green to red  
      - B: from blue to yellow  

[Find more info about CIELAB at wikipedia ref](https://en.wikipedia.org/wiki/CIELAB_color_space)  

  1.3 Normalize and reshape the Data  

      - X: use L as input  
      - Y: use A and B as target  
      Normalize the Y by Y/128 before training  

#### 2. Build Model: CNN  
#### 3. Colorize the test images  
#### 4. Save model and weights  

### II. Colorize images with pre-trained model:  

run file: Colorization_loadModel.py  

The python file will load saved model and weights, then colorize the images in test folder.


### III. Colorize videos with pre-trained model:  

run file: Colorization_video.py  

The python file will load saved model and weights, then colorize the video.

##### Process logic:  

  - First, dissect video file into frames (images) and save them.  
  - Then colorize the images by using the pre-trained model.  
  - Last, put the frames together as a video.  
