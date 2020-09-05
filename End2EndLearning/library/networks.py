# This script is to specify different network architectures.

import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Input, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import layers
from keras import activations
from keras.layers.normalization import BatchNormalization


def net_lstm(netType, nFramesSample):
	net = Sequential()
	
	if netType == 4:	## one-to-one
		net.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
	else:			   ## many-to-one or many-to-many
		net.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(nFramesSample, 66, 200, 3)))
	
	net.add(TimeDistributed(Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Flatten()))
	
	if netType == 3: 	## many-to-many
		net.add(LSTM(100, return_sequences=True))
		net.add(TimeDistributed(Dense(1)))
	else:			   ## many-to-one or one-to-one
		net.add(LSTM(100))
		net.add(Dense(1))
		
	net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	return net


def create_nvidia_network(BN_flag, fClassifier, nClass):
	if BN_flag == 0:
		net = net_nvidia_1(fClassifier, nClass)
	elif BN_flag == 1:
		net = net_nvidia_BN(fClassifier, nClass)
	elif BN_flag == 2:
		net = net_nvidia_AdvProp(fClassifier, nClass)
	return net

		

def net_nvidia(fClassifier, nClass):
	mainInput = Input(shape=(66,200,3))
	x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x2 = Flatten()(x1)
	z = Dense(100, kernel_regularizer=l2(0.001), activation='elu')(x2)
	z = Dense(50,  kernel_regularizer=l2(0.001), activation='elu')(z)
	z = Dense(10,  kernel_regularizer=l2(0.001), activation='elu')(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = Dense(nClass, activation='softmax')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = Dense(1, activation='sigmoid')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = Dense(1)(z)
		net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	return net


def net_nvidia_1(fClassifier, nClass):
	mainInput = Input(shape=(66,200,3))
	x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x2 = Flatten()(x1)
	z = Dense(100, kernel_regularizer=l2(0.001))(x2)
	z = layers.Activation(activations.elu)(z)
	z = Dense(50,  kernel_regularizer=l2(0.001))(z)
	z = layers.Activation(activations.elu)(z)
	z = Dense(10,  kernel_regularizer=l2(0.001))(z)
	z = layers.Activation(activations.elu)(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = Dense(nClass, activation='softmax')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = Dense(1, activation='sigmoid')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = Dense(1)(z)
		net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

	print(net.summary())
	return net


def net_nvidia_BN(fClassifier, nClass, lr=1e-4):
	mainInput = Input(shape=(66,200,3))
	x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x2 = Flatten()(x1)
	z = Dense(100, kernel_regularizer=l2(0.001))(x2)
	z = BatchNormalization()(z)
	z = layers.Activation(activations.elu)(z)
	z = Dense(50,  kernel_regularizer=l2(0.001))(z)
	z = BatchNormalization()(z)
	z = layers.Activation(activations.elu)(z)
	z = Dense(10,  kernel_regularizer=l2(0.001))(z)
	z = BatchNormalization()(z)
	z = layers.Activation(activations.elu)(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = Dense(nClass, activation='softmax')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = Dense(1, activation='sigmoid')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = Dense(1)(z)
		net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['accuracy'])
	return net


def net_nvidia_AdvProp(fClassifier, nClass):
	image_input_1 = Input(shape=(66,200,3), name='images_1')  # Variable-length sequence of ints
	image_input_2 = Input(shape=(66,200,3), name='images_2')  # Variable-length sequence of ints

	lambda0 = Lambda(lambda x: x/127.5 - 1.0)
	conv1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	elu_c1 = layers.Activation(activations.elu)
	conv2 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	elu_c2 = layers.Activation(activations.elu)
	conv3 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	elu_c3 = layers.Activation(activations.elu)
	conv4 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
	elu_c4 = layers.Activation(activations.elu)
	conv5 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
	elu_c5 = layers.Activation(activations.elu)

	flat = Flatten()

	dense1 = Dense(100, kernel_regularizer=l2(0.001))
	elu_d1 = layers.Activation(activations.elu)
	dense2 = Dense(50,  kernel_regularizer=l2(0.001))
	elu_d2 = layers.Activation(activations.elu)
	dense3 = Dense(10,  kernel_regularizer=l2(0.001))
	elu_d3 = layers.Activation(activations.elu)
	dense4 = Dense(1)

	x1 = lambda0(image_input_1)
	x1 = conv1(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c1(x1)
	x1 = conv2(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c2(x1)
	x1 = conv3(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c3(x1)
	x1 = conv4(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c4(x1)
	x1 = conv5(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c5(x1)
	x1 = flat(x1)

	x1 = dense1(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_d1(x1)
	x1 = dense2(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_d2(x1)
	x1 = dense3(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_d3(x1)
	output1 = dense4(x1)

	x2 = lambda0(image_input_2)
	x2 = conv1(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c1(x2)
	x2 = conv2(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c2(x2)
	x2 = conv3(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c3(x2)
	x2 = conv4(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c4(x2)
	x2 = conv5(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c5(x2)
	x2 = flat(x2)

	x2 = dense1(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_d1(x2)
	x2 = dense2(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_d2(x2)
	x2 = dense3(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_d3(x2)
	output2 = dense4(x2)

	net = Model(inputs=[image_input_1, image_input_2], outputs=[output1, output2], name='Nvidia_AdvProp')
	net.compile(optimizer=Adam(lr=1e-4),
				  loss=["mse", 'mse'],
				  loss_weights=[1, 1], metrics=['accuracy'])
	return net

	
if __name__ == "__main__":
	print('\n')
	print("### This is the file specifying different network architectures. Please do not run it directly.")
	print('\n')
