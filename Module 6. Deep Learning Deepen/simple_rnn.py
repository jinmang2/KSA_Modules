# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation
from keras.utils import np_utils

import os

from keras.utils.vis_utils import plot_model

# sample text
sample =  'hihello'

char_set = list(set(sample))
char_dic = { w : i for i, w in enumerate(char_set)}
print(char_set, '\n', char_dic)

x_str = sample[:-1]
y_str = sample[1:]

data_dim = len(char_set)
timesteps = len(y_str)
num_classes = len(char_set)

print(data_dim, timesteps, num_classes)

x = [char_dic[c] for c in x_str]
y = [char_dic[c] for c in y_str]

# One-hot encoding
x = np_utils.to_categorical(x, num_classes=num_classes)
x = x.reshape(-1, x.shape[0], data_dim)
print(x.shape)

y = np_utils.to_categorical(y, num_classes=num_classes)
y = y.reshape(-1, y.shape[0], data_dim)
print(y.shape)

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, SimpleRNN

model = Sequential()
model.add(LSTM(num_classes,
               input_shape=(timesteps, data_dim),
               return_sequences=True))
model.add(TimeDistributed(Dense(num_classes)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y, epochs=5)

predictions = model.predict(x, verbose=0)

for i, prediction in enumerate(predictions):
    print(prediction)
    x_index = np.argmax(x[i], axis=1)
    x_str = [char_set[j] for j in x_index]
    print(x_index, ''.join(x_str))
    
    index = np.argmax(prediction, axis=1)
    result = [char_set[j] for j in index]
    print(index, ''.join(result))