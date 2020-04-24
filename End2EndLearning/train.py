### This script is the main training file.

import sys
sys.path.insert(0, 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/End2EndLearning/library/')

from learning import train_dnn



if __name__ == "__main__":
    
	data_root = 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/'
    
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
    ## NOTE: paths must have forward slash (/) character at the end
    
    # NVIDIA dataset 
# 	trainPath = data_root + 'NVIDIA/'
    
    # custom driving simulator dataset
# 	trainPath = data_root + 'training_simu_1/'
    
    # track 1 of Udacity dataset
# 	trainPath = data_root + 'Udacity/track1data/'
    
    # track 2 of Udacity dataset
	trainPath = data_root + 'Udacity/track2data/'
    
	netType = 1        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	train_dnn(trainPath, netType, flags, specs)
	
	

	
		

	
	
	
	
