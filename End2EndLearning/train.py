### This script is the main training file.

import sys
import os

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

from learning import train_dnn

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == "__main__":

	import argparse

    # Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN to detect cityscapes.')
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

	outputPath = data_root + 'udacityA_nvidiaB_results/'
	if args.output_path != None:
		outputPath = args.output_path

	if not os.path.exists(outputPath):
		os.mkdir(outputPath)

	print('Image folder: '+imagePath)
	print('Label file: '+labelPath)
	print('Output folder: '+outputPath)

	netType = 1        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	train_dnn(imagePath, labelPath, outputPath, netType, flags, specs)
	