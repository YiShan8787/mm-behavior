import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.models import model_from_json, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import csv
import os
from time import time
import matplotlib.pyplot as plt

from mds_dict import mds_dict

class mds_dl:
	def __init__(self):
		self.behaviors = ['arm_moving','boxing','jump','kick_leg','morning_exercise','raise','run','stride','swing']
		#self.directory = '/home/wt/RadHARex/Data/DopplerTrain/'
		self.directory = '/home/wt/RadHARex/Data/new_data/'
		#self.test_file = './output/test.npy'
		self.nd = 64
		self.time_domain_bins = 10
		self.p_test = .1				# Test samples percent
		self.p_validation = .2			# Validation samples percent
		self.checkpoint_model_path = './output/mds_cnn_model.h5'

	def load(self):
		ts=time()
		print('Loading database...')
		data_X = np.empty((0,self.nd*self.time_domain_bins), float)
		data_Y = np.empty((0), int)
		for behavior in self.behaviors:
			childdir = self.directory + behavior + '/csv/'
			for file in os.listdir(childdir):
				if file.endswith("_1.npy"):
					print('Loading '+file+'...')
					data = np.load(childdir+file)
					if file.startswith("x_"):
						data_X = np.append(data_X, data, axis=0)
					else: # y_
						data_Y = np.append(data_Y, data)
					print(data_X.shape, data_Y.shape)
		'''
		for root,dirs,files in os.walk(self.directory):
			for file in files:
				if file.endswith(".npy"):
					print('Loading '+file+'...')
					data = np.load(self.directory+file)
					if file.startswith("x_"):
						data_X = np.append(data_X, data, axis=0)
					else: # y_
						data_Y = np.append(data_Y, data)
					print(data_X.shape, data_Y.shape)
		'''

		data_X = data_X.reshape(-1, self.nd, self.time_domain_bins, 1)
		print(data_X.shape)
		classes = np.unique(data_Y)
		nClasses = len(classes)

		data_Y_one_hot = to_categorical(data_Y)
		print("one hot:", data_Y_one_hot.shape)

		#train_X, self.test_X, train_label, self.test_label = train_test_split(data_X, data_Y_one_hot, test_size=self.p_test, random_state=42)
		#self.train_X, self.valid_X, self.train_label, self.valid_label = train_test_split(train_X, train_label, test_size=self.p_validation, random_state=13)
		self.train_X, self.valid_X, self.train_label, self.valid_label = train_test_split(data_X, data_Y_one_hot, test_size=self.p_validation, random_state=42)
		print('Training data shape:', self.train_X.shape, self.train_label.shape)
		print('Validating data shape:', self.valid_X.shape, self.valid_label.shape)
		#print('Test data shape:', self.test_X.shape, self.test_label.shape)
		print('Total data shape:', data_X.shape, data_Y.shape)
		print('Total number of classes:', nClasses)
		print('Load complete.')
		print('Time cost:', time()-ts,'s')

	def cnn(self):
		self.load()
		ts=time()
		print('Training using CNN...')

		batch_size = 64
		epochs = 10
		num_classes = self.train_label.shape[1]

		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(self.nd, self.time_domain_bins, 1),padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D((2, 2),padding='same'))
		model.add(Dropout(0.05))
		model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		model.add(Dropout(0.05))
		model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=0.1))                  
		model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		model.add(Dropout(0.05))
		model.add(Flatten())
		model.add(Dense(128, activation='linear'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dropout(0.05))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

		checkpoint = ModelCheckpoint(self.checkpoint_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

		callbacks_list = [checkpoint]

		model.summary()

		model_train = model.fit(self.train_X, self.train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(self.valid_X, self.valid_label),callbacks=callbacks_list)

		#test_eval = model.evaluate(self.test_X, self.test_label, verbose=0)
		
		print(model_train.history)
		np.save('./output/trained_history.npy', model_train.history)
		print('Trained history saved.')
		#print('Test loss:', test_eval[0])
		#print('Test accuracy:', test_eval[1])
		print('Train complete.')
		print('Time cost:', time()-ts,'s')
		model.save('./output/mds_cnn_model.h5')
		print('Saved model to disk.')

		# 绘制训练 & 验证的准确率值
		'''
		plt.plot(model_train.history['accuracy'])
		plt.plot(model_train.history['val_accuracy'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.show()

		# 绘制训练 & 验证的损失值
		plt.plot(model_train.history['loss'])
		plt.plot(model_train.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.show()
		'''
	def model_predict(self):
		# json_file = open('./output/model.json', 'r')
		# loaded_model_json = json_file.read()
		# json_file.close()
		# loaded_model = model_from_json(loaded_model_json)
		# loaded_model.load_weights('./output/model.h5')
		loaded_model = load_model('./output/mds_cnn_model.h5')
		print('Loaded model from disk.')
		f = open("acc.txt", "w")
		
		

		gt = 0
		for behavior in self.behaviors:
			childdir = self.directory + behavior + '/csv/'
			test_data_X = np.empty((0,self.nd*self.time_domain_bins), float)
			test_data_Y = np.empty((0), int)
			for file in os.listdir(childdir):
				if file.endswith("_2.npy"):
					print('Loading '+file+'...')
					test_data = np.load(childdir+file)
					if file.startswith("x_"):
						test_data_X = np.append(test_data_X, test_data, axis=0)
					else: # y_
						test_data_Y = np.append(test_data_Y, test_data)
					print(test_data_X.shape, test_data_Y.shape)
			
		
			test_data_X = test_data_X.reshape(-1, self.nd, self.time_domain_bins, 1)
			print('Test data shape:',test_data_X.shape)
			#classes = np.unique(test_data_Y)
			#nClasses = len(classes)

			#test_data_Y_one_hot = to_categorical(test_data_Y)
			pred = loaded_model.predict(test_data_X)
			print(pred.shape)
			pred_array   = []
			true_positive = 0
			all_number = pred.shape[0]
			for i,number in enumerate(pred):
				idx = np.argmax(number)
				pred_array.append(idx)
				if idx == gt:
					true_positive = true_positive+1
			print("acc:",true_positive/all_number)
			f.write(str(true_positive/all_number) + "\n")
			gt=gt+1
		f.close()
			
		print('finish testing.')

	def main(self):
		self.cnn()
		self.model_predict()

if __name__ == '__main__':
	mds_dl().main()