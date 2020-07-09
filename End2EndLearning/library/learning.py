# This script is for learning.

import cv2  
import os
import shutil
import numpy as np

import keras
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utilities import resize_image, random_distort, load_train_data, load_train_data_multi
from networks import net_lstm, net_nvidia, create_nvidia_network
import time
import ntpath



def gen_train_data_lstm_m2m(xList, yList, batchSize, nFramesSample):
	## get feature dimensions
	featureSample = resize_image(cv2.imread(xList[0]))
	
	## specify X and y shapes
	X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
	y = np.empty((batchSize, nFramesSample, 1))
	
	## generate training data
	sampleCount = 0
	while True:  
		for i in range(0, len(yList) - nFramesSample):
			## create a sample that has multiple frames
			for k in range(nFramesSample):
				X[sampleCount,k] = resize_image(cv2.imread(xList[i + k]))
				y[sampleCount,k] = yList[i + k]
				
			sampleCount += 1
				
			## yield a batch when #samples = batchSize
			if(sampleCount == batchSize):
				yield (X, y)
				X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
				y = np.empty((batchSize, nFramesSample, 1))
				sampleCount = 0
				
				
				
def gen_train_data_lstm_m2o(xList, yList, batchSize, nFramesSample):
	## get feature dimensions
	featureSample = resize_image(cv2.imread(xList[0]))
	
	## specify X and y shapes
	X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
	y = np.empty((batchSize, 1))
	
	## generate training data
	sampleCount = 0
	while True:  
		for i in range(0, len(yList) - nFramesSample):
			## create a sample that has multiple frames
			for k in range(nFramesSample):
				X[sampleCount,k] = resize_image(cv2.imread(xList[i + k]))
				
			y[sampleCount] = yList[i + nFramesSample]
			sampleCount += 1
				
			## yield a batch when #samples = batchSize
			if(sampleCount == batchSize):
				yield (X, y)
				X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
				y = np.empty((batchSize, 1))
				sampleCount = 0
					
						
def gen_train_data_random(xList, yList, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	while True:       
		for i in range(len(yList)):
			image_path = xList[i]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			angle = yList[i]
			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)
			X.append(img)
			y.append(angle)
			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				yield (np.array(X), np.array(y))
				X, y = ([],[])
				xList, yList = shuffle(xList, yList)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)

def gen_train_data_random_AdvProp(xList, yList, xList_advp, yList_advp, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	X_advp, y_advp = ([],[])
	while True:       
		for i in range(min(len(yList), len(yList_advp))):
			image_path = xList[i]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			angle = yList[i]

			image_path_advp = xList_advp[i]
			if not os.path.isfile(image_path_advp):
				image_path_advp = image_path_advp.replace(".jpg", "_fake.png")
			img_advp = resize_image(cv2.imread(image_path_advp))
			angle_advp = yList_advp[i]

			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)
				img_advp, angle_advp = random_distort(img_advp, angle_advp)


			#X.append([img, img_advp])
			#y.append([angle, angle_advp])
			X.append(img)
			y.append(angle)
			X_advp.append(img_advp)
			y_advp.append(angle_advp)

			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				#yield (np.array(X), np.array(y))
				#yield (np.array([X, X_advp]), np.array([y, y_advp]))
				yield [np.array(X), np.array(X_advp)], [np.array(y), np.array(y_advp)]
				X, y = ([],[])
				X_advp, y_advp = ([],[])
				xList, yList = shuffle(xList, yList)
				xList_advp, yList_advp = shuffle(xList_advp, yList_advp)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)


'''
def train_dnn(imageDir, labelPath, outputPath, netType, flags, specs):
	
	## assigning variables
	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
	batchSize 	   = specs[0]
	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	## prepare the data
	xList, yList = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)
	
	## change the data format if necessary
	if fClassifier:
		print('\n######### Classification #########')
		yTrainList = to_categorical(yTrainList, num_classes = nClass)
		yValidList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
	
	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = net_nvidia(fClassifier, nClass)
		trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
		validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2o(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2o(xValidList, yValidList, batchSize, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2m(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2m(xValidList, yValidList, batchSize, nFramesSample)

	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	else:
		shutil.rmtree(outputPath)
		os.makedirs(outputPath)
	modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
	lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
	
	## train
	nTrainStep = int(len(yTrainList)/batchSize) + 1
	nValidStep = int(len(yValidList)/batchSize) + 1
	net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
	verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)
	net.save(outputPath + 'model-final.h5')
	print(net.summary())
'''


def train_dnn_multi(imageDir_list, labelPath_list, outputPath, netType, flags, specs, modelPath = "", 
	trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imageDir_list_advp=[], labelPath_list_advp=[], reinitBN = False):
	
	## assigning variables
	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
	batchSize 	   = specs[0]
	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	## prepare the data
	xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio)

	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)

	if BN_flag == 2:
		xList_advp, yList_advp = load_train_data_multi(imageDir_list_advp, labelPath_list_advp, nRep, fThreeCameras, trainRatio)

		xTrainList_advp, xValidList_advp = train_test_split(np.array(xList_advp), test_size=0.1, random_state=42)
		yTrainList_advp, yValidList_advp = train_test_split(np.array(yList_advp), test_size=0.1, random_state=42)
	
	## change the data format if necessary
	if fClassifier:
		print('\n######### Classification #########')
		yTrainList = to_categorical(yTrainList, num_classes = nClass)
		yValidList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
	
	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
		if BN_flag <= 1:
			trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
			validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
		elif BN_flag == 2:
			trainGenerator = gen_train_data_random_AdvProp(xTrainList, yTrainList, xTrainList_advp, yTrainList_advp, batchSize)
			validGenerator = gen_train_data_random_AdvProp(xValidList, yValidList, xValidList_advp, yValidList_advp, batchSize)

	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2o(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2o(xValidList, yValidList, batchSize, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2m(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2m(xValidList, yValidList, batchSize, nFramesSample)

	if modelPath != "":
		print("pretrain modelPath: ", modelPath)
		net.load_weights(modelPath)
		print(partialPreModel)
		if partialPreModel:
			print("partial PreModel activate")
			#net_untrain = net_nvidia(fClassifier, nClass)
			start_layer_id=8
			for i in range(start_layer_id):
				net.layers[i].trainable = False
			#for i in range(start_layer_id, len(net.layers)):
				#net.layers[i].set_weights(net_untrain.layers[i].get_weights())
			#	net.layers[i].trainable = False
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
		if reinitHeader:
			print("reinit header activate")
			net_untrain = create_nvidia_network(BN_flag, fClassifier, nClass)
			net.layers[-1].set_weights(net_untrain.layers[-1].get_weights())
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
		if reinitBN:
			net_untrain = create_nvidia_network(BN_flag, fClassifier, nClass)
			BN_layer_ids = [3, 6, 9, 12, 15, 19, 22, 25]
			for id in BN_layer_ids:
				net.layers[id].set_weights(net_untrain.layers[id].get_weights())
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['accuracy'])


	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	else:
		shutil.rmtree(outputPath)
		os.makedirs(outputPath)
	modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
	lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
	
	## train
	nTrainStep = int(len(yTrainList)/batchSize) + 1
	nValidStep = int(len(yValidList)/batchSize) + 1
	net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
	verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)

	net.save(outputPath + 'model-final.h5')
	print(net.summary())
	
	
	
def train_nv_icra19(trainPath, trainSet, repSet, outputPath, batchSize, nEpoch):
	
	## prepare the data
	xList = [];
	yList = [];
	for i in range(len(repSet)):
		print(trainPath + trainSet[i] + '/')
		xTmp, yTmp = load_train_data(trainPath + trainSet[i] + '/', repSet[i], False)
		xList = xList + xTmp;
		yList = yList + yTmp;
		
	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)
	
	print('\n######### Regression #########')
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
		
	outputPath = outputPath + 'output/models-cnn/';
	net = net_nvidia(False, -1)
	trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
	validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
	
	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	else:
		shutil.rmtree(outputPath)
		os.makedirs(outputPath)
	modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
	lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
	
	## train
	nTrainStep = int(len(yTrainList)/batchSize)
	nValidStep = int(len(yValidList)/batchSize)
	net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
	verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)
	#net.save(outputFolder + 'model-final.h5')
	print(net.summary())

	
	
	
'''
def train_dnn_overfitting(trainSpec, xTrainList, yTrainList, xValidList, yValidList):	
	## assign variables
	outputFolder = trainSpec[0]
	batchSize    = trainSpec[1]
	nEpoch       = trainSpec[2]
	isClassify   = trainSpec[3]
	nClass       = trainSpec[4]
	randomDistortFlag = False
	
	## pulling out 128 random samples and training just on them, to make sure the model is capable of overfitting
	tmpIndices = np.random.randint(0, len(xTrainList), 128)
	xTrainList = xTrainList[tmpIndices]
	yTrainList = yTrainList[tmpIndices]
	
	tmpIndices = np.random.randint(0, len(xValidList), 12)
	xValidList = xValidList[tmpIndices]
	yValidList = yValidList[tmpIndices]
	
	
	X,y = ([],[])  
	for i in range(len(yTrainList)):
		img = resize_image(cv2.imread(xTrainList[i]))
		angle = yTrainList[i]
		X.append(img)
		y.append(angle)
	X = np.array(X)
	y = np.array(y)
	
	X_val, y_val = ([],[]) 
	for i in range(len(yValidList)):
		img = resize_image(cv2.imread(xValidList[i]))
		angle = yValidList[i]
		X_val.append(img)
		y_val.append(angle)
	X_val = np.array(X_val)
	y_val = np.array(y_val)
		
		
	## get a network
	net = net_testing(isClassify, nClass)

	## initialize generators
	if isClassify:
		print('\n######### Classification #########')
		trainLabelList = to_categorical(yTrainList, num_classes = nClass)
		validLabelList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('############################\n')
		

	# Fit the model
	history = net.fit(X, y, epochs=10, verbose=2, validation_data = (X_val,y_val), shuffle = True)
	net.save(outputFolder + 'model-final.h5')
	#print(net.summary())

'''

	
def test_dnn(modelPath, imageDir, labelPath, outputPath, netType, flags, specs, BN_flag=0, pathID=0):
	
    ## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
    
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	testFeatures, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	testFeatures = np.array(testFeatures)
	testLabels = np.array(testLabels)

    
	print(testFeatures)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')
	testData = []
	for i in range(len(testLabels)):
		image_path = testFeatures[i]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")
		img = resize_image(cv2.imread(image_path))
		testData.append(img)

	testData = np.array(testData)

    ## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
    ## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input                                           # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
	
	### predict and output
	if BN_flag <= 1:
		layer_outs = functor(testData)
		predictResults = layer_outs[-1]
	else:
		layer_outs = functor([testData, testData])
		predictResults = layer_outs[-2+pathID]
	#predictResults = net.predict(testData)
	#score, acc = net.evaluate(testData, testLabels)

	BN_means = []
	BN_stds = []
	for layer_id in range(len(layer_outs)):
		#if layer_id not in [1, 3, 5, 7, 9]:
		#	continue

		layer_out = layer_outs[layer_id]
		#print(layer_out.shape)
		if layer_id <= last_conv_id:
			for i in range(layer_out.shape[3]):
				BN_means.append(np.mean(layer_out[:,:,:,i]))
				BN_stds.append(np.std(layer_out[:,:,:,i]))
		else:
			BN_means.append(np.mean(layer_out[:,:]))
			BN_stds.append(np.std(layer_out[:,:]))

			
	f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_means.txt"),'w')
	#print(BN_means)
	#print(BN_stds)
	for mean in BN_means:
		f_BN.write("{:.5f}\n".format(mean))
	f_BN.close()
	f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_stds.txt"),'w')
	for std in BN_stds:
		f_BN.write("{:.5f}\n".format(std))
	f_BN.close()

	f = open(outputPath,'w')
	mse_loss = np.mean(np.square(predictResults.flatten() - testLabels))
	print("mse loss: " + str(mse_loss))
	f.write("mse loss: {:.5f}\n".format(mse_loss))

	#thresh_holds = [0.01, 0.033, 0.1, 0.33, 1, 3.3]
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
	#thresh_holds = [1, 2, 4, 8]
	acc_list = []
	for thresh_hold in thresh_holds:
		acc = np.sum(np.abs(predictResults.flatten() - testLabels) < thresh_hold) / len(testLabels)
		print("accuracy (+-" + str(thresh_hold) + "): " + str(acc))
		f.write("accuracy (+-{:.3f}): {:.5f}\n".format(thresh_hold, acc))
		acc_list.append(acc)

	print("mean accuracy: " + str(np.mean(acc_list)))
	f.write("mean accuracy: {:.5f}\n\n".format(np.mean(acc_list)))
    
	f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
    
	for p in range(len(predictResults)):
# 		if fClassifier:
#  			f.write(str(np.argmax(p)))
#  			print(np.argmax(p))
# 		else: 
        # for regression
		imgName = os.path.basename(testFeatures[p])
		prediction = predictResults[p][0]
		groundTruth = testLabels[p]
		f.write("{:^12.3f} {:^12.3f} {:^12.3f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
		f.write('\n')
	f.close()

# 	for i in range(len(testLabels)):
# 		print([str('%.4f' % float(j)) for j in predictResults[i]])

			
	print('********************************************')
	print('\n\n\n')


def read_float_list(file_name):
    x = []
    file_in = open(file_name, 'r')
    for y in file_in.read().split('\n'):
        if len(y) > 0:
            x.append(float(y))
    return x

def is_similar(val1, val2, val_thresh, percent_thresh):
	val_diff = abs(val1 - val2)
	percent_diff1 = abs(val_diff / val1)
	percent_diff2 = abs(val_diff / val2)

	if val_diff < val_thresh or (percent_diff1 < percent_thresh and percent_diff2 < percent_thresh):
		return True

	return False


def filter_dataset(modelPath, imageDir, labelPath, outputPath, netType, flags, specs, BN_flag=0, target_BN_folder="", filter_percent=0.1):
	

    ## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
    
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	testFeatures, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	testFeatures = np.array(testFeatures)
	testLabels = np.array(testLabels)

    ## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
    ## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input                                           # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
		

	print(testFeatures)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')
	f = open(outputPath,'w')
	similar_img_count = 0
	diff_score_list = []
	for id in range(len(testLabels)):
		image_path = testFeatures[id]
		label_value = testLabels[id]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")
		img = resize_image(cv2.imread(image_path))
		testData=[img]

		testData = np.array(testData)

		### predict and output
		if BN_flag <= 1:
			layer_outs = functor(testData)
			predictResults = layer_outs[-1]
		else:
			layer_outs = functor([testData, testData])
			predictResults = layer_outs[-2]
		#predictResults = net.predict(testData)
		#score, acc = net.evaluate(testData, testLabels)


		BN_means = []
		BN_stds = []
		for layer_id in range(len(layer_outs)):
			#if layer_id not in [1, 3, 5, 7, 9]:
			#	continue

			layer_out = layer_outs[layer_id]
			if layer_id <= last_conv_id:
				for i in range(layer_out.shape[3]):
					BN_means.append(np.mean(layer_out[:,:,:,i]))
					BN_stds.append(np.std(layer_out[:,:,:,i]))
			else:
				BN_means.append(np.mean(layer_out[:,:]))
				BN_stds.append(np.std(layer_out[:,:]))

		BN_means_target = read_float_list(target_BN_folder + "/BN_means.txt")
		BN_stds_target = read_float_list(target_BN_folder + "/BN_stds.txt")

		SINGLE_FEATURE_PERCENT_THRESH = 1
		SINGLE_FEATURE_VALUE_THRESH = 3
		MODEL_PERCENT_THRESH = 0.5

		similar_feature_count = 0
		for i in range(len(BN_means)):
			mean_value_diff = abs(BN_means[i] - BN_means_target[i])
			mean_percent_diff = abs((BN_means[i] - BN_means_target[i])/BN_means_target[i])
			#print("base mean: ", BN_means_target[i], "  current mean: ", BN_means[i], "  mean_value_diff: ", mean_value_diff, "  mean_percent_diff: ", mean_percent_diff, "\n")
			
			std_value_diff = abs(BN_stds[i] - BN_stds_target[i])
			std_percent_diff = abs((BN_stds[i] - BN_stds_target[i])/BN_stds_target[i])
			#print("base std: ", BN_stds_target[i], "  current std: ", BN_stds[i], "  std_value_diff: ", std_value_diff, "  std_percent_diff: ", std_percent_diff, "\n")
			
			if is_similar(BN_means_target[i], BN_means[i], SINGLE_FEATURE_VALUE_THRESH, SINGLE_FEATURE_PERCENT_THRESH) and is_similar(BN_stds_target[i], BN_stds[i], SINGLE_FEATURE_VALUE_THRESH, SINGLE_FEATURE_PERCENT_THRESH):
				similar_feature_count += 1

		BN_means = np.array(BN_means)
		BN_means_target = np.array(BN_means_target)
		BN_stds = np.array(BN_stds)
		BN_stds_target = np.array(BN_stds_target)

		BN_mean_diff = BN_means - BN_means_target
		BN_std_diff = BN_stds - BN_stds_target

		'''
		mean_all_percent_1 = np.linalg.norm(BN_mean_diff) / np.linalg.norm(BN_means_target)
		mean_all_percent_2 = np.linalg.norm(BN_mean_diff) / np.linalg.norm(BN_means)
		std_all_percent_1 = np.linalg.norm(BN_std_diff) / np.linalg.norm(BN_stds_target)
		std_all_percent_2 = np.linalg.norm(BN_std_diff) / np.linalg.norm(BN_stds)

		#print("mean_all_percent_1 ", mean_all_percent_1, "mean_all_percent_2 ", mean_all_percent_2, "  std_all_percent_1 ", std_all_percent_1, "  std_all_percent_2 ", std_all_percent_2)
		
		# for honda
		MEAN_PERCENTAGE_THRESH = 1.2
		STD_PERCENTAGE_THRESH = 0.5

		# for udacity
		MEAN_PERCENTAGE_THRESH = 1.2
		STD_PERCENTAGE_THRESH = 0.8
		if mean_all_percent_1 < MEAN_PERCENTAGE_THRESH and mean_all_percent_2 < MEAN_PERCENTAGE_THRESH and std_all_percent_1 < MEAN_PERCENTAGE_THRESH and std_all_percent_2 < STD_PERCENTAGE_THRESH:
			f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
			similar_img_count += 1
		'''
		'''
		print(id, " ", similar_feature_count / len(BN_means))
		if similar_feature_count / len(BN_means) > MODEL_PERCENT_THRESH:
			f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
			similar_img_count += 1
		'''

		mean_std_ratio = np.linalg.norm(BN_means_target) / np.linalg.norm(BN_stds_target)
		diff_score = np.linalg.norm(BN_mean_diff) + mean_std_ratio * np.linalg.norm(BN_std_diff)
		#print("mean diff: ", np.linalg.norm(BN_mean_diff), "  std diff: ", np.linalg.norm(BN_std_diff), "  total: ", diff_score)
		#print("mean: ", np.linalg.norm(BN_means), "  std: ", np.linalg.norm(BN_stds))
		diff_score_list.append((id, diff_score, np.linalg.norm(BN_means), np.linalg.norm(BN_stds)))

	f.close()
		
	#print(diff_score_list)
	diff_score_list = sorted(diff_score_list, key=lambda diff_score: diff_score[1])
	#print(diff_score_list)

	f = open(outputPath,'w')
	for i in range(int(len(testLabels)*filter_percent)):
		id = diff_score_list[i][0]
		image_path = testFeatures[id]
		label_value = testLabels[id]
		#f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
		f.write(os.path.basename(image_path) + ",,," + str(label_value) + "," +str(diff_score_list[i][2]) + "," +str(diff_score_list[i][3])+"\n")
	f.close()

	print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	print('similar_img_count ', int(len(testLabels)*filter_percent))
	print('total_count ', len(testLabels))
	print('ratio ', filter_percent)
	print('\n\n')

	
if __name__ == "__main__":
	print('\n')
	print("### This is the library file for testing. Please do not directly run it.")
	print('\n')
