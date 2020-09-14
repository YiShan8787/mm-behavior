"""
Bidirectional LSTMS on VOXELS

Usage:

- extract_path is the where the extracted data samples are available.
- checkpoint_model_path is the path where to checkpoint the trained models during the training process


EXAMPLE: SPECIFICATION

extract_path = '/Users/sandeep/Research/Ti-mmWave/data/extract/Train_Data_voxels_'
checkpoint_model_path="/Users/sandeep/Research/Ti-mmWave/data/extract/LSTM"
"""

extract_path = '/home/wt/RadHAR/Data/extract/Train_Data_voxels_'
checkpoint_model_path="/home/wt/RadHAR/Data/model"


import glob
import os
import numpy as np
# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
#from tensorflow import set_random_seed
#set_random_seed(rand_seed)
import tensorflow
tensorflow.random.set_seed(rand_seed)

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Activation
from keras.layers.core import Permute, Reshape
from keras import backend as K

from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Bidirectional,TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import load_model


sub_dirs=['stand','swing']

def one_hot_encoding(y_data, sub_dirs, categories=5):
    Mapping=dict()

    count=0
    for i in sub_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        Type = Type.decode('utf-8')
        lab=Mapping[Type]
        #lab = Mappin.get(Type)
        
        y_features2.append(lab)
    
    y_features=np.array(y_features2)
    #print(y_features.shape)
    y_features=y_features.reshape(y_features.shape[0],1)
    
    from keras.utils import to_categorical
    y_features = to_categorical(y_features)
    #print(y_features.shape)
    return y_features


def full_3D_model(summary=False):
    print('building the model ... ')
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=False, stateful=False,input_shape=(60, 10*1024) )))
    model.add(Dropout(.5,name='dropout_1'))
    model.add(Dense(128, activation='relu', name='DENSE_1'))
    model.add(Dropout(.5,name='dropout_2'))
    model.add(Dense(2, activation='softmax', name = 'output'))

    return model



frame_tog = [60]


#loading the train data
Data_path = extract_path+'stand'

data = np.load(Data_path+'.npz')
train_data = data['arr_0']
train_data = np.array(train_data,dtype=np.dtype(np.int32))
train_label = data['arr_1']
#print(train_label.shape)
del data
print(train_data.shape,train_label.shape)

Data_path = extract_path+'swing'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)


del data

print(train_data.shape,train_label.shape)




train_label = one_hot_encoding(train_label, sub_dirs, categories=2)

train_data = train_data.reshape(train_data.shape[0],train_data.shape[1], train_data.shape[2]*train_data.shape[3]*train_data.shape[4])

print('Training Data Shape is:')
print(train_data.shape,train_label.shape)



X_train, X_val, y_train, y_val  = train_test_split(train_data, train_label, test_size=0.20, random_state=1)
del train_data,train_label

##shuffle before use validation split
from sklearn.utils import shuffle
#np.random.shuffle(X_train)
#y_train[X_train[:,0]]
X_train, y_train = shuffle(X_train, y_train)

model = full_3D_model()


print("Model building is completed")


adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                       decay=0.0, amsgrad=False)

model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=adam,
                  metrics=['accuracy'])

checkpoint = ModelCheckpoint(checkpoint_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

import time

timestamp = time.time()

# Training the model
learning_hist = model.fit(X_train, y_train,
                             batch_size=20,
                             epochs=30,
                             verbose=1,
                             shuffle=True,
                             validation_split = 0.2,
                           #validation_data=(X_val,y_val),
                             
                           callbacks=callbacks_list
                          )

finish_timestamp = time.time()

#calculate time

timestruct = time.localtime(finish_timestamp - timestamp)
print(time.strftime('%Y-%m-%d %H:%M:%S', timestruct))

##show the history
import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_val, y_val, batch_size=20)
print("test loss, test acc:", results)

## Saving the model

model.save( checkpoint_model_path + '/LSTM.h5')   # HDF5 file, you have to pip3 install h5py if don't have it

del model