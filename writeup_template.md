# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/coldrainy/miniature-doodle/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. The histogram shows that the training data,validation data and testing data have the similar distribution.

![alt text](/source/download.png)
![alt text](/source/train.png)
![alt text](/source/validation.png)

The visualization of the data set and it's corresponding lable indicate that the data lable is matched.
![alt text](/source/1.png)
### Design and Test a Model Architecture
Data preprocess

First, I convert the images to grayscale because it is the shape which decide the meaning of the Traffic sign.The color here is the redundant information.So,i move the color information from the image.Then, I normalized the picture by using the image data subtracting 128,and then the subtraction divided by 128.This is because the normolization can reduce the amound of calculation and make the data comparable.

Here is an example of a traffic sign image before and after grayscaling.

![alt text](/source/4.png)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU		|   									|
|  Max pooling				|  2x2 stride,  outputs   5x5x16					|
|		flatten				|												|
|			Fully connected			|		inputs 400, outputs 120										|
|			RELU			|												|
|			drop out			|												|
|				Fully connected				|		inputs 120, outputs 84										|
|			RELU			|												|
|			drop out			|												|
|				Fully connected				|		inputs 84, outputs 43										|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer,my batch size is 64 and the epochs is 30,learning rate is 0.0017.The mu and sigma used for randomly initialize the weiths and biases are 0.002 and 0.11

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Well,in the beginning,I just copy the architechture of the LeNet to this project.But the accuracy of the model can only get 90 percent.Then I have tried to add more layers to the architechture.But it seems that it doesn't help too much.So I gave up this way.Then I focus on tuning the parameters.There are mainly three parameters I tried to tune,mu,sigma,and learning rate.After tuning the parameter,the accuracy of the validwild animals crossingation can sometimes reach 93 percent.But it was not stable.So I add drop out after the fully connected layer.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 97.2% 
* test set accuracy of 94.4%
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text](/web_data/1.jpg) ![alt text](/web_data/2.jpg) ![alt text](/web_data/3.jpg) 
![alt text](/web_data/4.jpg) ![alt text](/web_data/6.jpg) ![alt text](/web_data/8.jpg)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing      		| Wild animals crossing   									| 
| Turn right ahead     			| Turn right ahead		|
| Speed limit(60km/h)					| Speed limit(60km/h)											|
| Turn left ahead	      		| Turn left ahead	 				|
| Speed limit(30km/h)			| Speed limit(50km/h)				|
| No passing for vehicles over 3.5 metric tons			| No passing for vehicles over 3.5 metric tons	|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This accuracy is low than the test accuracy.It might because the data is too few.The result will change a lot even only one picture was classified failed.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a wild animals crossing sign (probability of 1), and the image does contain a  wild animals crossing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| wild animals crossing   									| 
| .0     				| Speed limit (20km/h) 										|
| .0					| Speed limit (30km/h)										|
| .0	      			| Speed limit (50km/h)					 				|
| .0				    | Speed limit (60km/h)      							|


For the second image, the model is relatively sure that this is a turn right ahead sign (probability of 1), and the image does contain a  turn right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| turn right ahead   									| 
| .0     				| Speed limit (20km/h) 										|
| .0					| Speed limit (30km/h)										|
| .0	      			| Speed limit (50km/h)					 				|
| .0				    | Speed limit (60km/h)      							|

For the third image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.     | Speed limit (30km/h)   						| 
| .0     | Speed limit (20km/h) 								|
| .0					| Speed limit (30km/h)								 |
| .0	    | Speed limit (50km/h)					 			|
| .0				 | Speed limit (70km/h)      			|
For the fouth image, the model is relatively sure that this is a Turn left ahead sign (probability of 1), and the image does contain a  Turn left ahead. The top five soft max probabilities were

| Probability         	|     Prediction	             | 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Turn left ahead   									| 
| .0     				| Speed limit (20km/h) 										|
| .0					| Speed limit (30km/h)										|
| .0	      			| Speed limit (50km/h)					 				|
| .0				    | Speed limit (60km/h)      							|

For the fifth image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 1), and the image does contain a  Speed limit (60km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.     | Speed limit (50km/h)   							| 
| .0     | Speed limit (20km/h) 								 |
| .0					| Speed limit (30km/h)										|
| .0	    | Speed limit (60km/h)					 				|
| .0				 | Speed limit (70km/h)      				|
For the sixth image, the model is relatively sure that this is aNo passing for vehicles over 3.5 metric tons sign (probability of 1), and the image does contain a  No passing for vehicles over 3.5 metric tons. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| No passing for vehicles over 3.5 metric tons   									| 
| .0     				| Speed limit (20km/h) 										|
| .0					| Speed limit (30km/h)										|
| .0	      			| Speed limit (50km/h)					 				|
| .0				    | Speed limit (60km/h)      							|
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


