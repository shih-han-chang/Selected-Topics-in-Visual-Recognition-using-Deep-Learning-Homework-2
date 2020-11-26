# Selected-Topics-in-Visual-Recognition-using-Deep-Learning-Homework-2
The target of homework 2 is Street View House Numbers digit detector and the dataset include 33,402 train images, 13,068 test images.

## Hardware   
  Use Linux with PyTorch to train this model  

## Dataset 
  * The training data have 33,402 images with a label file in matlab format.
  * The test data have 13,068 image. 
  
## Training SSD
  * First download the fc-reduced VGG-16 PyTorch base network weights at weights dir
  * To train SSD using the train script simply specify the parameters listed in train.py as a flag or manually change them.  
    - python train.py
## Test SSD
  * To evaluate a trained network:  
    - python test.py
  * Then, it will generate a.json file with test result
  
  
