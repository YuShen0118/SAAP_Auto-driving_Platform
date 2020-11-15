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
import tensorflow as tf
import keras


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


def create_nvidia_network(BN_flag, fClassifier, nClass, nChannel=3, Maxup_flag=False):
	if BN_flag == 0:
		return net_nvidia_1(fClassifier, nClass, nChannel, Maxup_flag)
	elif BN_flag == 1:
		return net_nvidia_BN(fClassifier, nClass)
	elif BN_flag == 2:
		return net_nvidia_AdvProp(fClassifier, nClass)
	elif BN_flag == 3:
		return net_nvidia_Add(fClassifier, nClass)

	#default
	return net_nvidia_1(fClassifier, nClass, nChannel)

		
'''
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
'''

def mean_accuracy(y_true, y_pred):
	
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]

	res_list = []
	for thresh_hold in thresh_holds:
		res_list.append(tf.math.reduce_mean(tf.cast(keras.backend.abs(y_true-y_pred) > thresh_hold, tf.float32)))

	MA = tf.math.reduce_mean(res_list)
	
	return MA
	

class MaxupModel(keras.Model):

	def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
		x, y = data
		adsf

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
			id = np.argmax(np.abs(y_pred-y), axis=-1)
			print(np.abs(y_pred-y))
			print(id)
			asdf
			loss = self.compiled_loss(y[id], y_pred[id], regularization_losses=self.losses)

        # Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
        # Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}


def net_nvidia_1(fClassifier, nClass, nChannel=3, Maxup_flag=False):
	mainInput = Input(shape=(66,200,nChannel))
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
		if Maxup_flag:
			net = MaxupModel(inputs = mainInput, outputs = mainOutput)
		else:
			net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
		#net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

	# print(net.summary())
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


class ShiftLayer(keras.layers.Layer):
	def __init__(self):
		super(ShiftLayer, self).__init__()

	def call(self, inputs):
		#x, mean1, std1, mean2, std2 = inputs
		x1, x2 = inputs
		mean1 = keras.backend.mean(x1)
		std1 = keras.backend.std(x1)

		mean2 = keras.backend.mean(x2)
		std2 = keras.backend.std(x2)

		x2 = tf.math.subtract(x2, mean2)
		x2 = tf.math.divide(x2, std2)
		x2 = tf.math.multiply(x2, std1)
		x2 = tf.math.add(x2, mean1)
		return x2

# class MeanLayer(keras.layers.Layer):
# 	def __init__(self):
# 		super(MeanLayer, self).__init__()

# 	def call(self, inputs):
# 		return keras.backend.mean(inputs)


# class StdLayer(keras.layers.Layer):
# 	def __init__(self):
# 		super(StdLayer, self).__init__()

# 	def call(self, inputs):
# 		return keras.backend.std(inputs)

def net_nvidia_Add(fClassifier, nClass):
	image_input_1 = Input(shape=(66,200,3), name='images_1')  # Variable-length sequence of ints
	image_input_2 = Input(shape=(66,200,3), name='images_2')  # Variable-length sequence of ints


	lambda0 = Lambda(lambda x: x/127.5 - 1.0)

	conv11 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	conv12 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	conv13 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	conv14 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
	conv15 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))

	elu1 = layers.Activation(activations.elu)
	elu2 = layers.Activation(activations.elu)
	elu3 = layers.Activation(activations.elu)
	elu4 = layers.Activation(activations.elu)
	elu5 = layers.Activation(activations.elu)

	conv21 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	conv22 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	conv23 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	conv24 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
	conv25 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))

	flat = Flatten()

	dense1 = Dense(100, kernel_regularizer=l2(0.001))
	dense2 = Dense(50,  kernel_regularizer=l2(0.001))
	dense3 = Dense(10,  kernel_regularizer=l2(0.001))
	dense4 = Dense(1)

	elu6 = layers.Activation(activations.elu)
	elu7 = layers.Activation(activations.elu)
	elu8 = layers.Activation(activations.elu)

	shift = ShiftLayer()
	# mean = MeanLayer()
	# std = StdLayer()



	x1 = lambda0(image_input_1)
	x1 = elu1(conv11(x1))
	x1 = elu2(conv12(x1))
	x1 = elu3(conv13(x1))
	x1 = elu4(conv14(x1))
	x1 = elu5(conv15(x1))

	# mean1 = keras.backend.mean(x1)
	# std1 = keras.backend.std(x1)
	# mean1 = mean(x1)
	# std1 = std(x1)



	x2 = lambda0(image_input_2)
	x2 = elu1(conv21(x2))
	x2 = elu2(conv22(x2))
	x2 = elu3(conv23(x2))
	x2 = elu4(conv24(x2))
	x2 = elu5(conv25(x2))

	# mean2 = keras.backend.mean(x2)
	# std2 = keras.backend.std(x2)
	# mean2 = mean(x2)
	# std2 = std(x2)
	# x2 = tf.math.subtract(x2, mean2)
	# x2 = tf.math.divide(x2, std2)
	# x2 = tf.math.multiply(x2, std1)
	# x2 = tf.math.add(x2, mean1)
	#x3 = shift([x2, mean1, std1, mean2, std2])
	x2 = shift([x1, x2])
	#x2 = shift_layer([x2, x2])
	#x2 = shift_layer(x2)

	x1 = flat(x1)

	x1 = elu6(dense1(x1))
	x1 = elu7(dense2(x1))
	x1 = elu8(dense3(x1))
	output1 = dense4(x1)

	x2 = flat(x2)

	x2 = elu6(dense1(x2))
	x2 = elu7(dense2(x2))
	x2 = elu8(dense3(x2))
	output2 = dense4(x2)

	# from keras.utils import plot_model 
	# plot_model(model, to_file='model.png')

	net = Model(inputs=[image_input_1, image_input_2], outputs=[output1, output2], name='Nvidia_ImageAndFeature')
	netI = Model(inputs=image_input_1, outputs=output1, name='Nvidia_ImageOnly')
	netF = Model(inputs=image_input_2, outputs=output2, name='Nvidia_FeatureOnly')

	from keras.utils import plot_model 
	plot_model(net, to_file='model.png')

	net.compile(optimizer=Adam(lr=1e-4),
				  loss=['mse', 'mse'],
				  loss_weights=[1, 1], metrics=['accuracy'])
	netI.compile(optimizer=Adam(lr=1e-4),
				  loss='mse', metrics=['accuracy'])
	netF.compile(optimizer=Adam(lr=1e-4),
				  loss='mse', metrics=['accuracy'])
	return net, netI, netF




'''
class Gaussian_noise_layer(layers.Layer):
	def __init__(self, initializer="he_normal", **kwargs):
		super(Gaussian_noise_layer, self).__init__(**kwargs)
		self.initializer = keras.initializers.get(initializer)

	def build(self, input_shape):
		self.std = self.add_weight(
			shape=[1],
			initializer=self.initializer,
			name="std",
			trainable=True,
		)

	def call(self, inputs):
		noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=self.std*1000, dtype=tf.float32) 
		return inputs + noise
'''

class Gaussian_noise_layer(keras.layers.Layer):
	def __init__(self):
		super(Gaussian_noise_layer, self).__init__()
		w_init = tf.random_normal_initializer()
		initial_value=w_init(shape=(1,1), dtype="float32")
		self.std = tf.Variable(initial_value=initial_value,trainable=True)
		print(self.std)
		print('!!!!!!!!!!!!!!!!!!!')

	def call(self, inputs):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
		print(self.std.eval(session=tf.compat.v1.Session()))
		print('!!!!!!!!!!!!!!!!!!!')
		noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=tf.reduce_sum(self.std).eval(session=tf.compat.v1.Session())[0], dtype=tf.float32) 
		return inputs + noise

'''
class Gaussian_noise_layer(keras.layers.Layer):
	def __init__(self):
		units=32
		input_dim=32
		super(Gaussian_noise_layer, self).__init__()
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(
			initial_value=w_init(shape=(input_dim, units), dtype="float32"),
			trainable=True,
		)
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(
			initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
		)

	def call(self, inputs):
		return tf.matmul(inputs, self.w) + self.b
'''

class GAN_Nvidia():
	def __init__(self):
		self.img_rows = 66
		self.img_cols = 200
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

        #optimizer = Adam(0.0002, 0.5)
		optimizer = Adam(lr=1e-4)

        # Build and compile the discriminator
		self.d = self.build_discriminators()
		self.d.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        # Build the generator
		self.g = self.build_generators()

        # The generator takes noise as input and generated imgs
		z = Input(shape=self.img_shape)
		img_gene = self.g(z)

        # For the combined model we will only train the generators
		self.d.trainable = False

        # The valid takes generated images as input and determines validity
		valid = self.d(img_gene)

        # The combined model  (stacked generators and discriminators)
        # Trains generators to fool discriminators
		self.combined = Model(z, valid)
		self.combined.compile(optimizer=optimizer, loss=self.generator_loss, metrics=['accuracy'])

	def generator_loss(self, y_true, y_pred):
		mse = tf.keras.losses.MeanSquaredError()
		return -mse(y_true, y_pred)

	def gaussian_noise_layer(self, input_layer, std):
		noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
		return input_layer + noise

	def build_generators(self):

		mainInput = Input(shape=self.img_shape)
		mainOutput = Gaussian_noise_layer()(mainInput)

		model = Model(inputs = mainInput, outputs = mainOutput)

		print("********************** Generator Model *****************************")
		model.summary()

		return model

	def build_discriminators(self):

		mainInput = Input(shape=self.img_shape)

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

		mainOutput = Dense(1)(z)

		model = Model(inputs = mainInput, outputs = mainOutput)
		print("********************** Discriminator Model *****************************")
		model.summary()

		return model



	
if __name__ == "__main__":
	print('\n')
	print("### This is the file specifying different network architectures. Please do not run it directly.")
	print('\n')
