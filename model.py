#########################
### Paramaters to set ###
#########################
save_path = 'model_cut_py.h5'
save_path_hist = 'model_cut_py.json'
crop_top = 60
crop_bottom = 20

################
### Imports ###
################
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input, Cropping2D, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import VGG16

#################
### Load data ###
#################
driving_log = pd.read_csv('data/driving_log.csv')


def update_path(source_path, new_path_prefix):
    filename = source_path.split('/')[-1]
    return new_path_prefix + filename


num_in = len(driving_log)
X_train = np.zeros((6 * num_in, 160, 320, 3), dtype=np.uint8)  # set up array in advance to save memory
y_train = np.zeros(6 * num_in, dtype=float)

for i, (img_path_orig, steering_angle) in enumerate(zip(driving_log['center'], driving_log['steering'])):
    img_path = update_path(img_path_orig, 'data/IMG/')
    image = plt.imread(img_path)
    X_train[2 * i] = image
    y_train[2 * i] = steering_angle
    X_train[2 * i + 1] = np.fliplr(image)
    y_train[2 * i + 1] = -steering_angle

for i, (img_path_orig, steering_angle) in enumerate(zip(driving_log['left'], driving_log['steering'])):
    img_path = update_path(img_path_orig, 'data/IMG/')
    image = plt.imread(img_path)
    steering_angle += 0.25  # correct for view from left side
    X_train[2 * num_in + 2 * i] = image
    y_train[2 * num_in + 2 * i] = steering_angle
    X_train[2 * num_in + 2 * i + 1] = np.fliplr(image)
    y_train[2 * num_in + 2 * i + 1] = -steering_angle

for i, (img_path_orig, steering_angle) in enumerate(zip(driving_log['right'], driving_log['steering'])):
    img_path = update_path(img_path_orig, 'data/IMG/')
    image = plt.imread(img_path)
    steering_angle -= 0.25  # correct for view from right side
    X_train[4 * num_in + 2 * i] = image
    y_train[4 * num_in + 2 * i] = steering_angle
    X_train[4 * num_in + 2 * i + 1] = np.fliplr(image)
    y_train[4 * num_in + 2 * i + 1] = -steering_angle

####################
### Set up model ###
####################

stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)

checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)


# custom loss function to put higher emphasis on large steering angles for which we have relatively little data
def high_value_emphasizing_loss(y_true, y_pred):
    weighted_squared_difference = (y_true - y_pred) ** 2 * (1 + 100 * np.abs(y_true))
    return weighted_squared_difference


# load pretrained network
pretrained = VGG16(weights='imagenet', include_top=False,
                   input_shape=(160 - crop_top - crop_bottom, 320, 3))
for layer in pretrained.layers:
    layer.trainable = False

# cut off the top four pretrained layers
# (when using pop the model could ne be saved. Hence, the method below.
# https://github.com/tensorflow/tensorflow/issues/22479)
pretrained_cut = Sequential()
for layer in pretrained.layers[:-4]:
    pretrained_cut.add(layer)

inp = Input(shape=(160, 320, 3))
x = Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)))(inp)
x = Lambda(lambda x: (x / 255.0) - 0.5)(x)
x = pretrained_cut(x)
x = Flatten()(x)
x = Dropout(rate=0.4)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(rate=0.2)(x) # higher dropout close to regression result seemed to lead to problems
x = Dense(100)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
model = Model(inputs=inp, outputs=x)

model.compile(loss=high_value_emphasizing_loss, optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=50,
                           batch_size=128, callbacks=[stopper, checkpoint])

import json

with open(save_path_hist, 'w') as fp:
    json.dump(history_object.history, fp)
