# AutomaticPersonalityBaseline

This is the baseline method for the competition. It takes the first frame of each video and put the images into a pretrained convolutional neural network caffe model, BLVC Reference Caffe Net, and then the results from second to last layer were extracted to be used in off the shelf Random Forest Regression from sklearn. 

To follow the steps, you will have to make sure you have the common libraries installed, as well as caffe and openCV (cv2). Caffe might prove to be a little challenging to install but you can use AWS image by searcing for CS280 in the publicly available AWS instance. 

Then the steps are as follows:

1. Use Video_to_img.ipynb to extract the first frame as a .jpg file from all the videos. See the Notebook file for specific instructions on image and video directory;

2. Use Baseline_Raw_data_collect.ipynb to run the images through said caffe CNN. You will first have to set the caffe root propoerly in the second cell. It should be rather straightforward and if there's any issue please refer to caffe website. The caffemodel will be downloaded automatically if you have it already;

3. Then modify the image path for train and validation images. The process will take a long time - 2 to 3 seconds for each image, on a 2012 Macbook Air with NO GPU. It might be dramatically faster if you use GPU. You can do so by replaceing caffe.set_mode_cpu() in cell 3 by caffe.set_mode_gpu().

4. You will get 2 pickle files, Validation_data.p and Raw_data6000.p, both of which can be found in this repo. They are both of a pandas dataframe, with format like 

    | VideoName           |E        |A        |C        |N        |O
851 |-9BZ8A9U7TE.002.mp4  |0.579439 |0.494505 |0.417476 |0.46875  |0.544444

5. Use Baseline.ipynb to generate the baseline results. This file contains comments and should be easy to read. 
