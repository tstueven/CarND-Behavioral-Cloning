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

model.save('model.h5')