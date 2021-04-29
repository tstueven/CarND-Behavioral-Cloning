# **Behavioral Cloning**

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"

[image2]: ./examples/placeholder.png "Grayscaling"

[image3]: ./examples/placeholder_small.png "Recovery Image"

[image4]: ./examples/placeholder_small.png "Recovery Image"

[image5]: ./examples/placeholder_small.png "Recovery Image"

[image6]: ./examples/placeholder_small.png "Normal Image"

[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by
executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline
I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In a way the model is very simple: A pretrained model for feature extraction and then a few dense layers for the
regression.

VGG16 was chosen as the pretrained model (model.py line 78-79) and is easily integrated as it is already part of the
Keras library. It consists of repetitions of two to three 3x3 convolutional layers which are then followed by max
pooling layers. Detailed information about the model can be found [here](https://arxiv.org/pdf/1409.1556.pdf).

The top four layers (3 convolutional, 1 max pooling) were cut off (model.py line 86-88). Then the output was flattened
and after a dropout layers followed by a dense layer with 256 output notes, and a ReLU activation function. Ot top of
this comes another dropout a second dense layer with 100 output nodes and another ReLU activation. connected to that is
the single output neuron.

#### 2. Attempts to reduce overfitting in the model

Taking a pretrained model and freezing the weights completely prevents overfitting in the feature extraction part of the
network.

In order to reduce overfitting the dense layers on top of the pretrained layers the model contains dropout layers (
model.py lines 95 + 98) as described above. The first dropout layer which is applied to the flattened output of the
cut-off pretrained network has a relatively high dropout rate of 0.4. The second one, after the first dense layer, has a
lower dropout rate of only 0.2.

The model was trained and validated on different data sets to ensure that the model was not overfitting. If the
validation loss didn't increase significantly for three epochs in a row training is sopped by callback. Also, the up to
this point best performing model gets saved. The model was tested by running it through the simulator and ensuring that
the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Unfortunately, I did not manage to play the simulator smoothly via VNC at all. Trying to get it to run locally led also
to a couple of problems. After quite a few frustrating hours, I opted to use only the data that were provided.

Luckily, the data turned out to be sufficient to get the model working well on the first track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the example of most networks dealing with image
analysis: Start with a bunch of convolutional layers, flatten those, and put some dense layers on top.

My first step was to use a pretrained convolution neural network model, cuf off the dense layers on top, and freeze the
weights. I thought this be appropriate because, even though the model was trained for other data, the convolutional
layers are responsible to extract general geometric features, and those my mode would need as well. The risk of
overfitting is also reduced by choosing the pretrained model since with the weights frozen a large part of the network
cannot over-fit as it just does not fit at all.

To investigate a few different pretrained models I put a simple dense structure on top and tested which performed
relatively well. The candidates I chose were VGG16, ResNet50 and InceptionV3. Of these, VGG16 delivered by far the best
performance. This was a little surprising regarding the ImageNet scores but maybe ResNet50 are InceptionV3 superior on
classification but lose the information where the classified object was localized. That information though is essential
for our regression task.

Then I concentrated on the dense layers on top and found that one layer with 256 output nodes with ReLU activation
followed by another dense layer with 128 output nodes and again ReLU before the final output node already led to pretty
good results.

To prevent overfitting I added a dropout layer with a rate of 0.4 after the pretrained model and one with a rate of 0.2
after the first dense layer. I tried to go higher here but results (i.e. validation loss) would always get worse. Also,
adding a dropout layer between the last dense layer and the output node proved problematic. It probably makes sense
though that an actual numerical value always needs als its inputs.

In order to gauge how well the model was working, I let keras split off 20 % of my image and steering angle data into a
validation set and trained on the other 80 %. Training seemed to be working pretty well.

The final step was to run the simulator to see how well the car was driving around track one. It started out well but
failed miserably in the sharper curves, i.e. in cases where the steering angle was supposed to be high. The reason for
this was likely to be that there were relatively few datapoint with a large steering angle compared to a mass of small
steering angle data.

The obvious solution should have been to get more data with a larger steering angle. I didn't manage to do that for
technical reasons mentioned above. I tried out augmentation then, mostly shearing and rotation, but that didn't work so
well either.

What finally solved the issue was to a self defined loss function (code line 72 - 74):

```python
def high_value_emphasizing_loss(y_true, y_pred):
    weighted_squared_difference = (y_true - y_pred) ** 2 * (1 + 100 * np.abs(y_true))
    return weighted_squared_difference
```

The idea was to put a higher emphasis on the data point with a relatively large steering angle. And it works! With the
multiplier (the 100 above) I experimented a little and found a value this high indeed beneficial.

At that point, the vehicle was able to drive autonomously around the track without leaving the road. It didn't look too
smooth though. Then I remembered that, when using a pretrained model with a small data set of different data, it is
recommended to cut off some convolutional layers as well. Cutting off the top four layers led indeed to better results.
But there were many more parameters to tune and training already took a lot longer. I tried cutting off another four
layers which led to a very slow training and did not seem to bring any benefits. Thus, I stuck to VGG16 without its
dense and minus for additional layers.

As well as the validation loss, the driving, too, seemed to look a little nicer ofter that cut off and the result can be
seen in the video.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-102) looks as follows:

| Layer (type)           |      Output Shape      |        Param #   |
| ------------- |:-------------:| -----:|
| input_2 (InputLayer)     |    (None, 160, 320, 3)   |    0         |
|cropping2d_1 (Cropping2D)  |  (None, 80, 320, 3)     |   0
|lambda_1 (Lambda)          |  (None, 80, 320, 3)     |  0
|sequential_1 (Sequential)  |  (None, 5, 20, 512)     |   7635264
|flatten_1 (Flatten)        |  (None, 51200)          |   0
|dropout_1 (Dropout)        |  (None, 51200)          |   0
|dense_1 (Dense)            |  (None, 256)            |   13107456
|activation_1 (Activation)  |  (None, 256)            |   0
|dropout_2 (Dropout)        |  (None, 256)            |   0
|dense_2 (Dense)            |  (None, 100)            |   25700
|activation_2 (Activation)  |  (None, 100)            |   0
|dense_3 (Dense)            |  (None, 1)              |   101

Total params: 20,768,521

Trainable params: 13,133,257

Non-trainable params: 7,635,264

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example
image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle
would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image
that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under
fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the
learning rate wasn't necessary.
