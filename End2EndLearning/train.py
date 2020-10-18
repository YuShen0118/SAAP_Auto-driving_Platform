### This script is the main training file.

import sys
import os
from test import test_network

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

from learning import train_dnn_multi

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train_network(imagePath, labelPath, outputPath, modelPath = "", trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imagePath_advp=[], labelPath_advp=[], trainRatio_advp = 1.0, reinitBN = False, classification = False, netType=1, Maxup_flag=False, pytorch_flag=False):
	train_network_multi([imagePath], [labelPath], outputPath, modelPath, trainRatio, partialPreModel, reinitHeader, BN_flag, 
		[imagePath_advp], [labelPath_advp], trainRatio_advp, reinitBN, classification, netType, pack_flag=False, Maxup_flag=Maxup_flag, pytorch_flag=pytorch_flag)

def train_network_multi(imagePath_list, labelPath_list, outputPath, modelPath = "", trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imagePath_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, classification = False, netType=1, pack_flag=False, Maxup_flag=False, pytorch_flag=False):
	print('Image folder: ' + str(imagePath_list))
	print('Label file: ' + str(labelPath_list))
	print('Output folder: ' + outputPath)

	if not os.path.exists(outputPath):
		os.mkdir(outputPath)

	## flags
	fRandomDistort = False
	fThreeCameras = False  # set to True if using Udacity data set
	fClassifier = classification
	flags = [fRandomDistort, fThreeCameras, fClassifier]
	
	## parameters
	batchSize = 128
	nEpoch = 1000
	nClass = 49        # only used if fClassifier = True
	nFramesSample = 5  # only used for LSTMs
	nRep = 1
	specs = [batchSize, nEpoch, nClass, nFramesSample, nRep]
	
	## train
    ## NOTE: paths must have forward slash (/) character at the end
    
	#netType = netType        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o, 5: GAN
	train_dnn_multi(imagePath_list, labelPath_list, outputPath, netType, flags, specs, modelPath, trainRatio, partialPreModel, reinitHeader, 
		BN_flag, imagePath_list_advp, labelPath_list_advp, trainRatio_advp, reinitBN, pack_flag, mid=0, Maxup_flag=Maxup_flag, pytorch_flag=pytorch_flag)

def get_suffix(level_id):
	'''
	if level_id == 1:
		return "_darker_2/"
	if level_id == 2:
		return "_darker/"
	if level_id == 3:
		return "_lighter/"
	if level_id == 4:
		return "_lighter_2/"
	'''
	if level_id <= 3:
		return "_darker_"+str(7-level_id*2)+"/"
	else:
		return "_lighter_"+str(level_id*2-7)+"/"


def get_suffix(level_id):
	if level_id <= 5:
		return "_darker_"+str(6-level_id)+"/"
	else:
		return "_lighter_"+str(level_id-5)+"/"


def get_new_level(channel_name, imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f):
	f.write('\nchannel_name: ' + channel_name + '\n')
	print('channel_name: ' + channel_name)
	MA_min = 1
	#for new_level in range(1,7):
	for new_level in range(1,11):
		imagePath = imagePath0 + '_' + channel_name + get_suffix(new_level)
		MA = test_network(modelPath, imagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
		print(new_level, ': ', MA)
		f.write(str(new_level) + ': ' + '{:.2f}'.format(MA) + '\n')
		if MA_min > MA:
			MA_min = MA
			level = new_level

	return level

def train_network_multi_factor_search(imagePath, labelPath, outputPath, modelPath = "", trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imagePath_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, classification = False, netType=1):
	print('Image folder: ' + str(imagePath))
	print('Label file: ' + str(labelPath))
	print('Output folder: ' + outputPath)

	if not os.path.exists(outputPath):
		os.mkdir(outputPath)

	## flags
	fRandomDistort = False
	fThreeCameras = False  # set to True if using Udacity data set
	fClassifier = classification
	flags = [fRandomDistort, fThreeCameras, fClassifier]
	
	## parameters
	nRound = 1000
	nEpoch = 1
	batchSize = 128
	nClass = 49        # only used if fClassifier = True
	nFramesSample = 5  # only used for LSTMs
	nRep = 1
	specs = [batchSize, nEpoch, nClass, nFramesSample, nRep]

	blur_level = 1
	noise_level = 1
	distortion_level = 1

	R_level = 1
	G_level = 1
	B_level = 1
	H_level = 1
	S_level = 1
	V_level = 1
	Y_level = 1
	U_level = 1
	V1_level = 1

	imagePath0 = imagePath[0:-1]

	val_ratio = 0.1
	f = open(outputPath+"factor_level_choices.txt",'w')
	for rid in range(nRound):
		print("round no: "+str(rid)+"\n")
		f.write("round no: "+str(rid)+"\n")
		blur_imagePath = imagePath0+'_blur_'+str(blur_level)+'/'
		noise_imagePath = imagePath0+'_noise_'+str(noise_level)+'/'
		distortion_imagePath = imagePath0+'_distort_'+str(distortion_level)+'/'
		#G_imagePath = imagePath0+'_G_darker/' if G_level == 1 else imagePath0+'_G_lighter/'
		#S_imagePath = imagePath0+'_S_darker/' if S_level == 1 else imagePath0+'_S_lighter/'
		#Y_imagePath = imagePath0+'_Y_luma_darker/' if Y_level == 1 else imagePath0+'_Y_luma_lighter/'
		R_imagePath = imagePath0 + '_R' + get_suffix(R_level)
		G_imagePath = imagePath0 + '_G' + get_suffix(G_level)
		B_imagePath = imagePath0 + '_B' + get_suffix(B_level)
		H_imagePath = imagePath0 + '_H' + get_suffix(H_level)
		S_imagePath = imagePath0 + '_S' + get_suffix(S_level)
		V_imagePath = imagePath0 + '_V' + get_suffix(V_level)
		Y_imagePath = imagePath0 + '_Y_luma' + get_suffix(Y_level)
		U_imagePath = imagePath0 + '_U_blueproj' + get_suffix(U_level)
		V1_imagePath = imagePath0 + '_V_redproj' + get_suffix(V1_level)

		#imagePath_list = [imagePath, blur_imagePath, noise_imagePath, distortion_imagePath, R_imagePath, G_imagePath, B_imagePath, \
		#					H_imagePath, S_imagePath, V_imagePath, Y_imagePath, U_imagePath, V1_imagePath]

		imagePath_list = [imagePath, blur_imagePath, noise_imagePath, distortion_imagePath, \
							R_imagePath, G_imagePath, B_imagePath, H_imagePath, S_imagePath, V_imagePath]

		#imagePath_list = [imagePath, blur_imagePath, noise_imagePath, distortion_imagePath, R_imagePath, G_imagePath, B_imagePath]
		#imagePath_list = [imagePath, S_imagePath]

		#Noise only
		#imagePath_list = [imagePath, imagePath0+'_noise_'+str(noise_level)+'/']
		#imagePath_list = [imagePath, distortion_imagePath]
		
		labelPath_list = [labelPath] * len(imagePath_list)


		train_dnn_multi(imagePath_list, labelPath_list, outputPath, netType, flags, specs, modelPath, trainRatio, partialPreModel, reinitHeader, 
			BN_flag, imagePath_list_advp, labelPath_list_advp, trainRatio_advp, reinitBN, pack_flag=False, mid=rid)

		modelPath = outputPath + "model-final.h5"
		valOutputPath = ""
		
		
		print('blur MAs:')
		f.write('\nblur MAs:\n')
		MA_min = 1
		#for new_blur_level in range(1,4):
		for new_blur_level in range(1,6):
			#blurImagePath = imagePath0+'_blur_'+str(new_blur_level*2-1)+'/'
			blurImagePath = imagePath0+'_blur_'+str(new_blur_level)+'/'
			MA = test_network(modelPath, blurImagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_blur_level, ': ', MA)
			f.write(str(new_blur_level) + ': ' + '{:.2f}'.format(MA) + '\n')
			if MA_min > MA:
				MA_min = MA
				blur_level = new_blur_level

		print('noise MAs:')
		f.write('\nnoise MAs:\n')
		MA_min = 1
		#for new_noise_level in range(1,4):
		for new_noise_level in range(1,6):
			#noiseImagePath = imagePath0+'_noise_'+str(new_noise_level*2-1)+'/'
			noiseImagePath = imagePath0+'_noise_'+str(new_noise_level)+'/'
			MA = test_network(modelPath, noiseImagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_noise_level, ': ', MA)
			f.write(str(new_noise_level) + ': ' + '{:.2f}'.format(MA) + '\n')
			if MA_min > MA:
				MA_min = MA
				noise_level = new_noise_level


		print('distort MAs:')
		f.write('\ndistort MAs:\n')
		MA_min = 1
		#for new_distort_level in range(1,4):
		for new_distort_level in range(1,6):
			#distortImagePath = imagePath0+'_distort_'+str(new_distort_level*2-1)+'/'
			distortImagePath = imagePath0+'_distort_'+str(new_distort_level)+'/'
			MA = test_network(modelPath, distortImagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_distort_level, ': ', MA)
			f.write(str(new_distort_level) + ': ' + '{:.2f}'.format(MA) + '\n')
			if MA_min > MA:
				MA_min = MA
				distortion_level = new_distort_level
		
		
		R_level = get_new_level('R', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		G_level = get_new_level('G', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		B_level = get_new_level('B', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)

		H_level = get_new_level('H', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		S_level = get_new_level('S', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		V_level = get_new_level('V', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		


		'''
		Y_level = get_new_level('Y_luma', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		U_level = get_new_level('U_blueproj', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		V1_level = get_new_level('V_redproj', imagePath0, modelPath, labelPath, valOutputPath, BN_flag, val_ratio, f)
		'''

		print('new blur level: ', blur_level)
		print('new noise level: ', noise_level)
		print('new distort level: ', distortion_level)
		print('new R channel level: ', R_level)
		print('new G channel level: ', G_level)
		print('new B channel level: ', B_level)
		print('new H channel level: ', H_level)
		print('new S channel level: ', S_level)
		print('new V channel level: ', V_level)
		#print('new Y channel level: ', Y_level)
		#print('new U channel level: ', U_level)
		#print('new V1 channel level: ', V1_level)
		f.write("\n")
		f.write("new blur level: "+str(blur_level)+"\n")
		f.write("new noise level: "+str(noise_level)+"\n")
		f.write("new distort level: "+str(distortion_level)+"\n")
		f.write("new R channel level: "+str(R_level)+"\n")
		f.write("new G channel level: "+str(G_level)+"\n")
		f.write("new B channel level: "+str(B_level)+"\n")
		f.write("new H channel level: "+str(H_level)+"\n")
		f.write("new S channel level: "+str(S_level)+"\n")
		f.write("new V channel level: "+str(V_level)+"\n")
		#f.write("new Y channel level: "+str(Y_level)+"\n")
		#f.write("new U channel level: "+str(U_level)+"\n")
		#f.write("new V1 channel level: "+str(V1_level)+"\n\n")
		f.write("\n")
		f.flush()
	f.close()


if __name__ == "__main__":

	import argparse

    # Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train CNN to predict steering angle.')
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

	outputPath = data_root + 'udacityA_nvidiaB_results/training_models/'
	if args.output_path != None:
		outputPath = args.output_path

	train_network(imagePath, labelPath, outputPath)
	