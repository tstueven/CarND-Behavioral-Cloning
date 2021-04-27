import numpy as np
import pandas as pd
from scipy import ndimage

def update_path(source_path, new_path_prefix):
    filename = source_path.split('/')[-1]
    return new_path_prefix + filename

driving_log = pd.read_csv('data/driving_log.csv')

images = []
measurements = driving_log['steering']

for img_path_orig in driving_log['center']:
    img_path = update_path(img_path_orig, 'data/IMG/')
    image = ndimage.imread(img_path)
    images.append(image)
    
X_train = np.array(images)
y_train = measurements

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')import numpy as np

import pandas as pd

from scipy import ndimage

import matplotlib.pyplot as plt

​

def update_path(source_path, new_path_prefix):

    filename = source_path.split('/')[-1]

    return new_path_prefix + filename

​

driving_log = pd.read_csv('data/driving_log.csv')

driving_log.head()

	center 	left 	right 	steering 	throttle 	brake 	speed
0 	IMG/center_2016_12_01_13_30_48_287.jpg 	IMG/left_2016_12_01_13_30_48_287.jpg 	IMG/right_2016_12_01_13_30_48_287.jpg 	0.0 	0.0 	0.0 	22.148290
1 	IMG/center_2016_12_01_13_30_48_404.jpg 	IMG/left_2016_12_01_13_30_48_404.jpg 	IMG/right_2016_12_01_13_30_48_404.jpg 	0.0 	0.0 	0.0 	21.879630
2 	IMG/center_2016_12_01_13_31_12_937.jpg 	IMG/left_2016_12_01_13_31_12_937.jpg 	IMG/right_2016_12_01_13_31_12_937.jpg 	0.0 	0.0 	0.0 	1.453011
3 	IMG/center_2016_12_01_13_31_13_037.jpg 	IMG/left_2016_12_01_13_31_13_037.jpg 	IMG/right_2016_12_01_13_31_13_037.jpg 	0.0 	0.0 	0.0 	1.438419
4 	IMG/center_2016_12_01_13_31_13_177.jpg 	IMG/left_2016_12_01_13_31_13_177.jpg 	IMG/right_2016_12_01_13_31_13_177.jpg 	0.0 	0.0 	0.0 	1.418236

images = []

steering_angles = []

​

for img_path_orig, steering_angle in zip(driving_log['center'], driving_log['steering']):

    img_path = update_path(img_path_orig, 'data/IMG/')

    image = plt.imread(img_path)

    images.append(image)

    steering_angles.append(steering_angle)

    images.append(np.fliplr(image))

    steering_angles.append(-steering_angle)

    

for img_path_orig, steering_angle in zip(driving_log['left'], driving_log['steering']):

    img_path = update_path(img_path_orig, 'data/IMG/')

    image = plt.imread(img_path)

    images.append(image)

    steering_angle += 0.25

    steering_angles.append(steering_angle)

    images.append(np.fliplr(image))

    steering_angles.append(-steering_angle)

    

for img_path_orig, steering_angle in zip(driving_log['left'], driving_log['steering']):

    img_path = update_path(img_path_orig, 'data/IMG/')

    image = plt.imread(img_path)

    images.append(image)

    steering_angle -= 0.25

    steering_angles.append(steering_angle)

    images.append(np.fliplr(image))

    steering_angles.append(-steering_angle)

    

X_train = np.array(images)

y_train = np.array(steering_angles)

X_train.shape

(48216, 160, 320, 3)

y_train.shape

(48216,)

from keras.applications import VGG19, VGG16, InceptionV3

​

pretrained = VGG19(weights='imagenet', include_top=False, input_shape=(160,320,3))

pretrained = VGG16(weights='imagenet', include_top=False, input_shape=(160,320,3))

#pretrained = InceptionV3(weights='imagenet', include_top=False, input_shape=(160,320,3))

​

for layer in pretrained.layers:

    layer.trainable = False

Using TensorFlow backend.
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])

WARNING:tensorflow:From /opt/miniconda3/envs/car2/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From /opt/miniconda3/envs/car2/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /opt/miniconda3/envs/car2/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

WARNING:tensorflow:From /opt/miniconda3/envs/car2/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/opt/miniconda3/envs/car2/lib/python3.5/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])

WARNING:tensorflow:From /opt/miniconda3/envs/car2/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /opt/miniconda3/envs/car2/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

from keras.models import Sequential, Model

from keras.layers import Flatten, Dense, Lambda, Input

​

​

inp = Input(shape=(160,320,3))

x = Lambda(lambda x: (x / 255.0) - 0.5)(inp)

x = pretrained(x)

x = Flatten()(x)

x = Dense(1)(x)

​

model = Model(inputs = inp, outputs=x)

# out = Dense(512, activation = 'relu')(out)

# model = Sequential()

# model.add(Lambda(normalize, input_shape=(160,320,3)))

# model.add(Flatten())

# model.add(Dense(1))

​

​

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, batch_size=100)

​

model.save('model.h5')