# This script is for learning.

import cv2  
import os
import shutil
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utilities import resize_image, random_distort, load_train_data
from networks import net_lstm, net_nvidia
import time



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

	
def test_dnn(modelPath, imageDir, labelPath, outputPath):
	
    ## assigning variables
# 	fRandomDistort = flags[0]
# 	fThreeCameras  = flags[1]
# 	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
# 	nClass         = specs[2]
# 	nFramesSample  = specs[3]
# 	nRep  = specs[4]
    
        
        
	print('\n\n\n')
	print('********************************************')
	
# 	if fClassifier:
# 		print('Classification......')
# 	else:
	print('Regression......')

	### retrieve the test data
	testFeatures, testLabels = load_train_data(imageDir, labelPath, nRep=1)
	testFeatures = np.array(testFeatures)
	testLabels = np.array(testLabels)

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
# 	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
	net = net_nvidia(False, 2)
# 	elif netType == 2:
# # 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
# 		net = net_lstm(2, nFramesSample)
# 	elif netType == 3:
# # 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
# 		net = net_lstm(3, nFramesSample)

#     ## load model weights
# 	if modelPath:
# 		net.load_weights(modelPath)
    
	### predict and output
	predictResults = net.predict(testData)
	score, acc = net.evaluate(testData, testLabels)
	print("Test loss: " + str(score))
	print("Test accuracy: " + str(acc))
    
	f = open(outputPath,'w')
	f.write("mse loss: {:.5f}\naccuracy: {:.5f}\n\n".format(score, acc))
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

	
if __name__ == "__main__":
	print('\n')
	print("### This is the library file for testing. Please do not directly run it.")
	print('\n')
