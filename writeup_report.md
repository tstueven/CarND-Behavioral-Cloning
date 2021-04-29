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

[center]: ./examples/center.jpg "Center"

[center_flipped]: ./examples/center_flipped.jpg "Center flipped"

[left]: ./examples/left.jpg "Left"

[left_flipped]: ./examples/left_flipped.jpg "Left flipped"

[right]: ./examples/left.jpg "Right"

[right_flipped]: ./examples/left_flipped.jpg "Right flipped"

[loss_evo]: ./examples/loss_evolution.png "Model evolution"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by
executing

```sh
python drive.py model.h5
```

There is a minor modification in the drive.py file to allow loading models that were saved with a custom loss function.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline
I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In a way, the model is very simple: A pretrained model for feature extraction and then a few dense layers for the
regression.

VGG16 was chosen as the pretrained model (model.py line 78-79) and is easily integrated as it is already part of the
Keras library. It consists of repetitions of two to three 3x3 convolutional layers which are then followed by max
pooling layers. Detailed information about the model can be found [here](https://arxiv.org/pdf/1409.1556.pdf).

The top four layers (3 convolutional, 1 max pooling) were cut off (model.py line 86-88). Then the output was flattened
and after that a dropout layers was followed by a dense layer with 256 output notes, and a ReLU activation function. On
top of this comes another dropout, a second dense layer with 100 output nodes and another ReLU activation. Connected to
that is the single output neuron.

At the beginning, even before the pretrained network, the data were cropped (model.py line 91) and normalized (line 92).

#### 2. Attempts to reduce overfitting in the model

Taking a pretrained model and freezing the weights prevents overfitting in the feature extraction part of the network
completely.

In order to reduce overfitting the dense layers on top of the pretrained layers, the model contains dropout layers (
model.py lines 95 + 98) as described above. The first dropout layer which is applied to the flattened output of the
cut-off pretrained network has a relatively high dropout rate of 0.4. The second one, after the first dense layer, has a
lower dropout rate of only 0.2.

The model was trained and validated on different data sets to ensure that the model was not overfitting. If the
validation loss didn't increase significantly for five epochs in a row, training is stopped by callback. Also, the up to
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
layers are responsible for extracting general geometric features, and those my mode would need as well. The risk of
overfitting is also reduced by choosing the pretrained model since, with the weights frozen, a large part of the network
cannot over-fit as it just does not fit at all.

To investigate a few different pretrained models I put a simple dense structure on top and tested which performed
relatively well. The candidates I chose were VGG16, ResNet50 and InceptionV3. Of those, VGG16 delivered by far the best
performance. This was a little surprising regarding the ImageNet scores but maybe ResNet50 are InceptionV3 superior on
classification but lose the information where the classified object is localized. That information though is essential
for our regression task.

Then I concentrated on the dense layers on top and found that one layer with 256 output nodes with ReLU activation
followed by another dense layer with 128 output nodes and again ReLU before the final output node already led to pretty
good results.

To prevent overfitting I added a dropout layer with a rate of 0.4 after the pretrained model and one with a rate of 0.2
after the first dense layer. I tried to go higher here but results (i.e. validation loss) would always get worse. Also,
adding a dropout layer between the last dense layer and the output node proved problematic. It probably makes sense
though that an actual numerical value always needs all its inputs.

In order to gauge how well the model was working, I let keras split off 20 % of my image and steering angle data into a
validation set and trained on the other 80 %. Training seemed to be working pretty well.

The final step was to run the simulator to see how well the car was driving around track one. It started out well but
failed miserably in the sharper curves, i.e. in cases where the steering angle should be high. The reason for this was
likely to be that there were relatively few datapoints with a large steering angle compared to a mass of small steering
angle data.

The obvious solution should have been to get more data with a larger steering angle. I didn't manage to do that for
technical reasons mentioned above. I tried out augmentation then, mostly shearing and rotation, but that didn't work so
well either.

What finally solved the issue was to use a self defined loss function (code line 72 - 74):

```python
def high_value_emphasizing_loss(y_true, y_pred):
    weighted_squared_difference = (y_true - y_pred) ** 2 * (1 + 100 * np.abs(y_true))
    return weighted_squared_difference
```

The idea was to put a higher emphasis on the data point with a relatively large steering angle. And it works! With the
multiplier (the 100 above) I experimented a little and found a value this high is indeed beneficial.

At that point, the vehicle was able to drive autonomously around the track without leaving the road. It didn't look too
smooth though. Then I remembered that, when using a pretrained model with a small data set of different data, it is
recommended to cut off some convolutional layers as well. Cutting off the top four layers led indeed to better results.
But there were many more parameters to tune and training already took a lot longer. I tried cutting off another four
layers which led to a very slow training and did not seem to bring any benefits. Thus, I stuck to VGG16 without its
dense and minus for additional layers.

As well as the validation loss, the driving, too, seemed to look a little nicer after that cut-off and the result can be
seen in the video.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-102) looks as follows:

| Layer (type)           |      Output Shape      |        Param #   |
| ------------- |:-------------:| -----:|
| input_2 (InputLayer)     |    (None, 160, 320, 3)   |    0         |
|cropping2d_1 (Cropping2D)  |  (None, 80, 320, 3)     |   0
|lambda_1 (Lambda)          |  (None, 80, 320, 3)     |  0
|vgg16_cut_off (Sequential)  |  (None, 5, 20, 512)     |   7635264
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

The layer vgg16_cut_off (Sequential) in the table above is a summary for the cut off pretrained VGG16 network with the
following structure:

|Layer (type)         |        Output Shape   |           Param #   |
| ------------- |:-------------:| -----:|
| input_1 (InputLayer)      |   (None, 80, 320, 3)    |    0|
|block1_conv1 (Conv2D)      |  (None, 80, 320, 64)   |    1792|
|block1_conv2 (Conv2D)      |  (None, 80, 320, 64)    |   36928|
|block1_pool (MaxPooling2D) |  (None, 40, 160, 64)    |   0|
|block2_conv1 (Conv2D)       | (None, 40, 160, 128)   |   73856|
| block2_conv2 (Conv2D)      |  (None, 40, 160, 128)  |    147584|
| block2_pool (MaxPooling2D) |  (None, 20, 80, 128)   |    0|
| block3_conv1 (Conv2D)      |  (None, 20, 80, 256)   |    295168|
| block3_conv2 (Conv2D)      |  (None, 20, 80, 256)   |    590080|
| block3_conv3 (Conv2D)      |  (None, 20, 80, 256)   |    590080|
| block3_pool (MaxPooling2D) |  (None, 10, 40, 256)   |    0|
| block4_conv1 (Conv2D)      |  (None, 10, 40, 512)    |   1180160|
| block4_conv2 (Conv2D)      |  (None, 10, 40, 512)  |     2359808|
| block4_conv3 (Conv2D)      |  (None, 10, 40, 512)  |     2359808|
| block4_pool (MaxPooling2D) |  (None, 5, 20, 512)   |     0|
| block5_conv1 (Conv2D)      |  (None, 5, 20, 512)   |     2359808|
| block5_conv2 (Conv2D)      |  (None, 5, 20, 512)   |     2359808|
| block5_conv3 (Conv2D)      |  (None, 5, 20, 512)   |     2359808|
| block5_pool (MaxPooling2D) |  (None, 2, 10, 512)   |     0    |

Total params: 14,714,688

Trainable params: 0

Non-trainable params: 14,714,688

0 trainable parameters tells us that the weights are frozen and will not be fitted for this part.

#### 3. Creation of the Training Set & Training Process

For aforementioned reasons I had to rely on the training data provided with the project. As far as I can see, they were
taken from one lap driven in the reversed direction.

There are images from three different camera positions: center, left, and right. Examples below:

![alt text][center]
![alt text][left]
![alt text][right]

For the left and right images, the steering angle had to be corrected for training since later only the center image is
used as input and it should work with that. The correction is done when loading the data (model.py lines 47 + 56). For
the left image the angle has to be increased, for the right decreased. 0.25 rad turned out to be a good correction
value.

To have a little more data available, especially some leftward curves as well, I flipped the images and added those to
the data as well. The steering angle was multiplied by -1 for these data.

![alt text][center_flipped]
![alt text][left_flipped]
![alt text][right_flipped]

That, in the end, led to 48216 data points, 38572 of which were randomly chosen for training, 9644 for validation. The
data fir comfortably in memory and thus I didn't use a generator.

I then started training for up to 50 epochs. I likely never need as many here but since there is an early stopping
callback in place there is no harm in choosing a higher number. The evolution of the training and validation loss can be
seen below:

![alt text][loss_evo]

The train loss starts at the very lage value (logscale!) of 1421.62. Since there are more than 13 M parameters to fit
that is probably not surprising. The validation loss at the end of that training cycle is already a lot lower, 6.68. The
validation continue to perform better for a while, likely because there is no dropout involved. Train and validation
loss drop quickly first, but especially the validation soon gets towards saturation. After the tenth epoch (starting at
zero like in plot), right before the training loss becomes better, the minimum is reached. At that point the model was
saved by a ModelCheckpoint callback and used for training. 