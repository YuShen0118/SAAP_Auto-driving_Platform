### This script is the main training file.

import sys
sys.path.insert(0, 'library/')

from learning import train_dnn



if __name__ == "__main__":

	## flags
	fRandomDistort = False
	fThreeCameras = False
	fClassifier = False
	flags = [fRandomDistort, fThreeCameras, fClassifier]
	
	## parameters
	batchSize = 128
	nEpoch = 1000
	nClass = 2        # only used if fClassifier = True
	nFramesSample = 5  # only used for LSTMs
	nRep = 1
	specs = [batchSize, nEpoch, nClass, nFramesSample, nRep]
	
	## train
	trainPath = 'D:/data/Kitti/object/training_simu_20200109/'
	netType = 1        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	train_dnn(trainPath, netType, flags, specs)
	
	

	
		

	
	
	
	
