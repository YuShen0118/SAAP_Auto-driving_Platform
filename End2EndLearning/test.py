### This script is the testing file for regression DNN.
### Implementation assumptions in test_dnn: 
    ## netType=1 (CNN), nRep=1, nClass=2, loss function = mse
    ## fThreeCameras=False, fClassifier=False

import sys
import os

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

from learning import test_dnn

def test_network(modelPath, imagePath, labelPath, outputPath):
	if modelPath:
		print('Model used: ' + modelPath)
	else:
		print('No model specified. Using random initialization of weights.')
        
	print('Image folder: '+imagePath)
	print('Label file: '+labelPath)
	print('Output file: '+outputPath)
	
	file_path = os.path.dirname(outputPath)
	if not os.path.exists(file_path):
		os.mkdir(file_path)

	## flags
	fRandomDistort = False
	fThreeCameras = False  # set to True if using Udacity data set
	fClassifier = False
	flags = [fRandomDistort, fThreeCameras, fClassifier]
	
	## parameters
	batchSize = 128
	nEpoch = 1000
	nClass = 2        # only used if fClassifier = True
	nFramesSample = 5  # only used for LSTMs
	nRep = 1
	specs = [batchSize, nEpoch, nClass, nFramesSample, nRep]
	
	netType = 1        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	test_dnn(modelPath, imagePath, labelPath, outputPath, netType, flags, specs)


if __name__ == "__main__":

	import argparse

    # Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Test Mask R-CNN to detect cityscapes.')
	parser.add_argument('--model_path', required=False,
						metavar="/path/to/image/folder/",
						help='/path/to/image/folder/')
	parser.add_argument('--image_folder_path', required=False,
						metavar="/path/to/image/folder/",
						help='/path/to/image/folder/')
	parser.add_argument('--label_file_path', required=False,
						metavar="/path/to/label/file",
						help="/path/to/label/file")
	parser.add_argument('--output_path', required=False,
						metavar="/path/to/output/folder/",
						help="/path/to/output/folder/")
	args = parser.parse_args()


	data_root = ROOT_DIR + '/Data/'
    
    ### NOTE: the only flags/parameters used in test are fClassifier and nClass. 
    ### Everything was included in test.py in case other models needed to be used in the future.
    
	## train
    ## NOTE: paths must have forward slash (/) character at the end
    
    # NVIDIA dataset 
	trainPath = data_root + 'udacityA_nvidiaB/'
    
    #image folder path
	imagePath = trainPath + 'trainB/'
	if args.image_folder_path != None:
		imagePath = args.image_folder_path

	#label file path
	labelPath = trainPath + 'labelsB_train.csv'
	if args.label_file_path != None:
		labelPath = args.label_file_path

    #model weight file path
	modelPath = None
	if args.label_file_path != None:
		modelPath = args.model_path
        
    # output path, should be a filename. 
	outputPath = data_root + 'udacityA_nvidiaB_results/test_results/test.txt'

	if args.output_path != None:
		outputPath = args.output_path

##  outputPath should be a filename, not directory
	if not os.path.exists(os.path.dirname(outputPath)):
		os.mkdir(os.path.dirname(outputPath))

	if modelPath:
		print('Model used: ' + modelPath)
	else:
		print('No model specified. Using random initialization of weights.')
        
	print('Image folder: '+imagePath)
	print('Label file: '+labelPath)
# 	print('Output file: '+outputPath)

# 	netType = 1        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	test_dnn(modelPath, imagePath, labelPath, outputPath)

