[//]: # (Image References)
[image0]: ./docs/misc/sim_screenshot.png
[image1]: ./docs/FCN_Model0.png
[image2]: ./docs/Following_Target_v2.png
[image3]: ./docs/Patrol_Without_Target_v2.png
[image4]: ./docs/Patrol_With_Target_v2.png
[image5]: ./docs/Following_Target_v3.png
[image6]: ./docs/Patrol_Without_Target_v3.png
[image7]: ./docs/Patrol_With_Target_v3.png
[image8]: ./docs/Training_Curves_32.png
[image9]: ./docs/Training_Curves_64.png

# Project: Follow Me Using Deep Learning 
## Deep Learning Project ##

In this project, we will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

In particular, we will apply the deep learning technique called Fully Convolutional Network (FCN) to the images captured by the cameras mounted onto the drone. The drone has the task of following its master. Since the drone is not as smart as the humans or the animals/birds, we need to train the drone using the images from the camera, to identify the master and follow. The master could be a person or an animal. In any case, the drone first needs to train using the appropriate image set and then predict.  If the master changes or master's characteristics changes then the drone's image network must be retrained to calibrate to the new master or charactersitics. For example, if the master changes its apprearance, then the network must be retrained in order to be able to execute the following mode. 

![alt text][image0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/DrPanigrahi/RoboND-DeepLearning.git
```

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

We'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

**Training Deep Learning Model**
The FCN model needs to be trained using the `model_training.ipynb`. The training process can be extremely time consuming on a local computer.  I have tried several parameter tuning in order to get the desired accuracy lavel of 40%.  Then using the trained model we can perform the segmentation and classification of the master target. Once we are comfortable with performance on the training dataset, we can check how it performs in live simulation!


## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow us to verify that our segmentation network is semi-functional. However, if we are interested in improving our score, we may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With our training and validation data having been generated or downloaded from the this repository, we are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training the Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

### Hyperparameters
Define and tune your hyperparameters.
- **batch_size**: number of training samples/images that get propagated through the network in a single pass.
- **num_epochs**: number of times the entire training dataset gets propagated through the network.
- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well.
- **workers**: maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with. 

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of errors are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. In addition to this we determine whether the network detected the target person or not. If more than 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. We determine whether the target is actually in the image by checking whether there are more thnn 3 pixels containing the target in the label mask. 

Using the above scoring mechanism, the number of detections for true positives, false positives and false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this or improve your network architecture and hyperparameters. Do not use the sample_evaluation_data for training the model.  If we use the sample_evaluation_data to train the network, it will result in inflated scores, and we will not be able to determine how our network will actually perform when evaluated with different set of data. 


## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`

## Final Results

### Network Architecture
The fully convolutional network (FCN) in this project cosists of the input layer, three/four encoder layers, one 1x1 convolution and three/four decoder layers. Each encoder layer reduces the size of the previous layer by a factor of two while increasing the depth of the convolution. The encoder layers consist of a convolution followed by a batch normalization layer which ensures faster learning progress by re-normalizing the feature vectors for each batch during training. The encoder layers are followed by a 1x1 convolution that retains the shape of the output of the last encoder. This convolutional layer is used instead of a fully connected layer in order to retain the spacial information of the segmented image. After this layer, three/four decoder layers are used to interpolate the segmentation back to the resolution of the input image. Each decoder uses bilinear interpolation to increase the resolution by a factor of two. 

In the network with the highest accuracy the first encoder has a filter depth of 64 and the 1x1 convolution has a filter depth of 256. The output layer consists of a dense layer with softmax activation function. It has three output classes color coded for background, target object (the master/hero) and other objects. Thus the fully-connected output layer will be able to classify each of the pixels in the image as one of the three output classes. The following figure shows the FCN model that performed the best. 

![alt text][image1] 

### Parameter Tuning
The hyperparameters used will determine not only how well the network gets trained but also determines how fast our network gets trained. 

After several trials with tuning the hyper parameters I have used the following fr=or the final results:
```python
number_of_training_images = 4131
number_of_validation_images = 1184

learning_rate = 0.002
batch_size = 48
num_epochs = 48
steps_per_epoch = number_of_training_images // batch_size + 1     #87
validation_steps = number_of_validation_images // batch_size + 1  #25
workers = 8

print("steps per epoch: ", steps_per_epoch)
print("validation steps: ", validation_steps)
```
Adam algorithm is used for the training optimization. The deafult learning for Adam optimizer in Keras is 0.001.  However, I chose 0.002 to speed up the training. We also have to take care to not make the leraning rate too high or too low. Too high learning rate might not be able to converge to the minimum and at times might even skip the minimum and blow up. Too low learining rate might result in the training getting stuck in a local minimum. This problem addressed using the Adam optimizer since it dynamically adjusts the training rate as the training progresses. 

Below table shows the accuracy score and training time on a local Ubuntu computer:

ecoders/decoders | filter_size | batch_size | num_epochs | workers | training_time | accuracy_score (%)
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | -------------
3/3 | 32 | 50 | 30 | 3 | 08 hrs. (15 min/epoch) | 36.23
3/3 | 32 | 50 | 40 | 3 | 10 hrs. (15 min/epoch) | 39.78
3/3 | 64 | 48 | 48 | 8 | 28 hrs. (35 min/epoch) | 43.21
4/4 | 32 | 48 | 40 | 8 | 20 hrs. (30 min/epoch) | 38.58

The training time heavily depends on the filter size. By doubling the filter size from 32 to 64 resulted in three times the training time.  Also, keeping the same filter size (32) but making deeper convolution layers, caused the training time to be longer.  However, the deeper model (4 encoders and 4 decoders) with smaller filter depth (32) seems to take lesser training time per epoch than the larger filter depth. Keeping all parameters the same but adding additional convolution layer caused the model training time to be doubles while it did not help improve the accuracy of prediction (compare row2 and row 4 in the above table). However it seems that increasing the filter depth from 32 to 64 and increasing the number of epochs helped improve the prediction accuracy. The final training loss for model trained with encoder size 32 (left, for second row from the table) and 64 (right, for third row from the table) are shown below.

![alt text][image8] 
![alt text][image9]

The examples for the target following, patroling without target and patroling with target is shown below for the model with ecoder size 32.

![alt text][image2]
![alt text][image3]
![alt text][image4]

The examples for the target following, patroling without target and patroling with target is shown below for the model with ecoder size 64.

![alt text][image5]
![alt text][image6]
![alt text][image7]


## Model Limitations and Possible Enhancements
### Model Limitations
- The model is trained to identify the simulated human model or the master in red outfit. 
- The same trained model can not be used to identify other human models in different colored outfit, nor could it be used to identify other type of animals or objects. 
- For indetification of other type of objects, the model needs to be re-trained before being used for prediction. 
- The model accuracy is quite low even for a simulation environment. So, this model might perform much worse in the real-world scenario and hence unusable! 

### Possible Enhancements
- It is possible to reduce the training time with a higher learning rate, however, it is hard to predict whether the model accuracy will be impoved. 
- In order to get better accuracy it would be worth invetigating by reducing the batch size while increasing the filter depth. 
- Perhapse by decreasing the filter depth by half for the first layer and then adding additional encoder and decoder layers to make even deeper model might be advantageous. 
- Decreasing the learning rate and increasing the number of epochs will also help improve the scores a bit, but would be hard to predict by how much the model will be improved. 
- Only 37.64% of the files contained the master (the humanoid with red dress). Increasing the number of images that contains the master might also increase the model accuracy.
- I would really like to use this FCN to train real-world images. I might try it out at a later time.

### Notes to Udacity Curriculum Developers
- When I took the course, I had an impression that I will be learning state of the art techniques!  However, the deeplearning methods covered in the course material seems completely outdated.  Am I supposed to celebrate for getting an accuracy score of 40%?  This 40% is not only completely useless in the simulation scenario but also getting the score of 40% will be absolutely impossible to acchive in the real world scnario with varying lighting and weather conditions. The 40% accuracy would have been state of the art a decade ago, not in the year 2018!  The goal of the project should have been to achieve at least 80% accuracy.  It is disappointing that Udacity abruptly ended the course without providing methods to improve the accuracy upto atleast 80%-90%. I would suggest that Udacity include addidtional deep learning techniques (for example GAN) to bring the accuracy scores above 80% in real-world conditions.
- It would be great if Udacity provided some real-world images for training and testing. I would suggest this as a proposed add-on/enhancement to this course/project for future students.


