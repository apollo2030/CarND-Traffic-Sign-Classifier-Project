# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/original_cropped.png  "Cropping"
[image2]: ./images/image_after_pre_processing.png "Grayscaling"
[image3]: ./images/example_count_per_class.png "Count per class"
[image4]: ./images/test_signs.png "Traffic Signs"
[image5]: ./images/conv1_activation_map.png "First convolutional layer activation map"
[image6]: ./images/conv2_activation_map.png "Second convolutional layer activation map"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python methods to compute some datasets characteristics:
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of the example counts in each group of data.

![Count of validation, testing and training examples][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I resized the images to their original size from the 'sizes' vector and cropped the images using the 'coords' vector whch then I resized back to the input size - 32 x 32 x 3.

As a second step, I decided to convert the images to grayscale because the characteristics of the traffic signs are distiguashable from the shape and pictograms on the signs alone, this also reduces the size of the network.

Here is an example of a traffic sign image before and after cropping.

![alt text][image1]

As a last step, I normalized the image data because it makes the features (which are in this case the pixel light intensity) more uniform as an input for the CNN.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|Layer (type)                 | Output Shape           |    Param #  | 
|:---------------------------:|:----------------------:|:-----------:|
| Input                       | (Grayscale, 32, 32, 1) |   1024      |
| conv2d_1 (Conv2D)           | (None, 30, 30, 7)      |   70        |
| max_pooling2d_1 (MaxPooling2| (None, 15, 15, 7)      |   0         |
| activation_1 (Activation)   | (None, 15, 15, 7)      |   0         |
| conv2d_2 (Conv2D)           | (None, 11, 11, 16)     |   2816      |
| max_pooling2d_2 (MaxPooling2| (None, 5, 5, 16)       |   0         |
| dropout_1 (Dropout)         | (None, 5, 5, 16)       |   0         |
| activation_2 (Activation)   | (None, 5, 5, 16)       |   0         |
| flatten_1 (Flatten)         | (None, 400)            |   0         |
| dense_1 (Dense)             | (None, 220)            |   88220     |
| dropout_2 (Dropout)         | (None, 220)            |   0         |
| activation_3 (Activation)   | (None, 220)            |   0         |
| dense_2 (Dense)             | (None, 120)            |   26520     |
| activation_4 (Activation)   | (None, 120)            |   0         |
| dense_3 (Dense)             | (None, 43)             |   5203      |
| activation_5 (Activation)   | (None, 43)             |   0         |

##### Summary
Total params: 122,829                                            
Trainable params: 122,829                                        
Non-trainable params: 0                                          
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the keras library that made it trivial to add remove and change the number of layers as well as the the number of parameters. I used 18 training epochs, with the adam optimizer,  'categorical_crossentropy' for the loss function, and optimize for accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9580
* validation set accuracy of 0.9560
* test set accuracy of 0.9341

First I tried with the uncropped 32x32x3 images for the training set which gave an accuracy of 80%, after many attempts of increasing the size of the convolutional layers and the fully connected layers, the maximum I got was 85%. A real breakthrough was after I cropped the images, it made the the accuracy go to 88%. After testing with fewer nodes in the network layers, I tried to use the grayscale images instead, which made an improvement bumping the accuracy to 90%. Later, I changed the normalization function which made the features range from -0.5 to 0.5 instead of the -1 to 1, this alone increased the accuracy to 92%, the final step was to add dropout layers which made the network output an accuracy of 95%.

I chose LeNet as the base architecture for the network, the main difference is that I increased the number of filters to 7 for the first convolutional layer, and added dropout layers after the second convolutional layer and after the first fully connected layer.

I learned about LeNet as a good network for clasifying the handwritten numbers, I saw similarities between the 2 problems - relatively small number of classes, (10 vs 43) and the low variance of the input data meaning that a traffic sign as well as a number have to have specific caracteristics to be recognized as such.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

From the 5 images only 1 is not represented in the initial data set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.


| Predicted                   | Actual                    | Certainty | Result           |
|:---------------------------:|:-------------------------:|:---------:|:----------------:|
|0 - Road narrows on the right| Road narrows on the right | 71.3604%  |:heavy_check_mark:|
|1 - No passing               | Keep left                 | 99.6103%  |:x:               |
|2 - Priority road            | Priority road             | 100.000%  |:heavy_check_mark:|
|3 - Keep right               | Keep right                | 99.9985%  |:heavy_check_mark:|
|4 - Road work                | Road work                 | 99.9960%  |:heavy_check_mark:|

### Visualizing the Neural Network 

Here is the activation map for the first convolutional layer:

![alt text][image5] 

As you can see, each filter picks up specific caracteristics as the outline of the pictogram of the sign, the white part in the center, effects of different light conditions.

Here is the activation map of the second convolutional layer:

![alt text][image6]

The features picked up at this level are more abstract, like diagonal lines and images with something in the center.


