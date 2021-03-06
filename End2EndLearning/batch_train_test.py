### This script is the main training file.

import sys
import os
import glob
import cv2
# cv2.setNumThreads(0)

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from train import train_network, train_network_multi, train_network_multi_factor_search
from test import test_network, test_network_multi, visualize_network_on_image
from library.fid_score import *
from networks_pytorch import create_nvidia_network_pytorch
from networks import create_nvidia_network

DATASET_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB/"
OUTPUT_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB_results/"
TRAIN_OUTPUT_ROOT = OUTPUT_ROOT + "train_results/"
TEST_OUTPUT_ROOT = OUTPUT_ROOT + "test_results/"

if not os.path.exists(OUTPUT_ROOT):
	os.mkdir(OUTPUT_ROOT)

if not os.path.exists(TRAIN_OUTPUT_ROOT):
	os.mkdir(TRAIN_OUTPUT_ROOT)

if not os.path.exists(TEST_OUTPUT_ROOT):
	os.mkdir(TEST_OUTPUT_ROOT)

def get_label_file_name(folder_name, suffix=""):
	pos = folder_name.find('_')
	if pos == -1:
		main_name = folder_name
	else:
		main_name = folder_name[0:pos]

	if "train" in folder_name:
		labelName = main_name.replace("train","labels") + "_train"
	elif "val" in folder_name:
		labelName = main_name.replace("val","labels") + "_val"

	labelName = labelName + suffix
	labelName = labelName + ".csv"
	return labelName

def single_test():
	train_folder = "trainB"
	val_folder = "valB"
	pytorch_flag = False

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
	#train_network(imagePath, labelPath, outputPath)

	if pytorch_flag:
		modelPath = outputPath + "/model-final.pth"
	else:
		modelPath = outputPath + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder, "")
	labelPath = DATASET_ROOT + labelName
	#labelPath = DATASET_ROOT + "labelsB_train.csv"

	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag)

def single_test_with_config(subtask_id=-1):
	train_folder = "trainB"
	val_folder = "valB"
	BN_flag = 0
	net_type = 1  # 1: CNN (default), 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o, 5: GAN
	#suffix = "_similarBN40"
	train_label_suffix = ""
	#val_label_suffix = "_small"
	val_label_suffix = ""
	classification = False
	visualize = False
	radius = 10 #only valid when visualize = True
	Maxup_flag = False
	pytorch_flag = False
	modelPath = ""
	suffix = ""

	if subtask_id == '0':
		train_folder = "trainB"
		val_folder = "valB"
	elif subtask_id == '1':
		train_folder = "trainC1"
		val_folder = "valC1"
	elif subtask_id == '2':
		train_folder = "trainB"
		val_folder = "valC1"
	elif subtask_id == '3':
		train_folder = "trainC1"
		val_folder = "valB"
	elif subtask_id == '4':
		train_folder = "trainA"
		val_folder = "valA"
	elif subtask_id == '5':
		train_folder = "trainA"
		val_folder = "valB"
	elif subtask_id == '6':
		train_folder = "trainB"
		val_folder = "valA"
	elif subtask_id == '7':
		# pytorch baseline
		pytorch_flag = True
		suffix = "_norm"
	elif subtask_id == '8':
		Maxup_flag = True
		pytorch_flag = True
		suffix = "_5aug"
	elif subtask_id == '9':
		Maxup_flag = True
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainB_Maxup_pytorch/model-final.pth"
		#suffix = "_test"
	elif subtask_id == '10':
		train_folder = "trainH"
		val_folder = "valH"
		pytorch_flag = True
		# modelPath = TRAIN_OUTPUT_ROOT + "trainB_Maxup_pytorch/model-final.pth"
		#suffix = "_test"
	elif subtask_id == '11':
		train_folder = "trainH"
		val_folder = "valH"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainH_pytorch/model-final.pth"
		#suffix = "_test"
	elif subtask_id == '12':
		train_folder = "trainB"
		val_folder = "valB"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainH_pytorch/model-final.pth"
		#suffix = "_test"
	elif subtask_id == '13':
		train_folder = "trainB0.1"
		val_folder = "valB0.1"
		pytorch_flag = True
		#modelPath = TRAIN_OUTPUT_ROOT + "trainH_pytorch/model-final.pth"
		#suffix = "_test"
	elif subtask_id == '14':
		train_folder = "trainHs"
		val_folder = "valHs"
		pytorch_flag = False
	elif subtask_id == '15':
		train_folder = "trainHcs"
		val_folder = "valHcs"
		pytorch_flag = True
	elif subtask_id == '16':
		train_folder = "trainAds"
		val_folder = "valAds"
		pytorch_flag = False
	elif subtask_id == '17':
		train_folder = "trainAds"
		val_folder = "valAds"
		pytorch_flag = True


	train_suffix = train_label_suffix
	if BN_flag > 0:
		train_suffix = train_suffix + "_BN" + str(BN_flag)
	if classification:
		train_suffix = train_suffix + "_classify"
	if net_type == 5:
		train_suffix = train_suffix + "_GAN"
	if Maxup_flag:
		train_suffix = train_suffix + "_Maxup"
	if pytorch_flag:
		train_suffix = train_suffix + "_pytorch"
	if modelPath != "":
		train_suffix = train_suffix + "_retrain"

	train_suffix = train_suffix + suffix

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder, train_label_suffix)
	labelPath = DATASET_ROOT + labelName

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + train_suffix + "/"
	train_network(imagePath, labelPath, outputPath, modelPath=modelPath, BN_flag=BN_flag, classification=classification, netType=net_type, Maxup_flag=Maxup_flag, pytorch_flag=pytorch_flag)

	modelPath = outputPath + "/model-final.h5"
	if pytorch_flag:
		modelPath = outputPath + "/model-final.pth"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder, val_label_suffix)
	labelPath = DATASET_ROOT + labelName
	#labelPath = DATASET_ROOT + "labelsB_train.csv"

	if visualize:
		outputPath = TEST_OUTPUT_ROOT + train_folder + train_suffix + "__" + val_folder + "__r" + str(radius) + "__visualize/test_result.txt"
	else:
		outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + train_suffix + ")_(" + val_folder + ")/test_result.txt"

	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=BN_flag, classification=classification, visualize=visualize, radius=radius, pytorch_flag=pytorch_flag)


def single_test_AdvProp():
	train_folder = "trainB"
	val_folder = "valB"
	train_folder_advp = "trainB_blur_5"
	val_folder_advp = "valB_blur_5"
	#suffix = "_similarBN40"
	suffix = ""
	pathID = 0

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	imagePath_advp = DATASET_ROOT + train_folder_advp + "/"
	labelName_advp = get_label_file_name(train_folder_advp, suffix)
	labelPath_advp = DATASET_ROOT + labelName_advp

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + train_folder_advp + suffix + "_advp/"
	#train_network(imagePath, labelPath, outputPath, BN_flag=2, imagePath_advp=imagePath_advp, labelPath_advp=labelPath_advp)

	modelPath = outputPath + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder)
	labelPath = DATASET_ROOT + labelName

	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + train_folder_advp + suffix + "_advp)_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=2, pathID=pathID)


def single_test_ImgAndFeature():
	train_folder = "trainB"
	val_folder = "valB"
	train_folder_add = "trainC1"
	val_folder_add = "valC1"
	#suffix = "_similarBN40"
	suffix = ""
	pathID = 0
	modelPath = ""
	pytorch_flag = True

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	imagePath_add = DATASET_ROOT + train_folder_add + "/"
	labelName_add = get_label_file_name(train_folder_add, suffix)
	labelPath_add = DATASET_ROOT + labelName_add

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + train_folder_add + suffix + "_add/"
	#modelPath = TRAIN_OUTPUT_ROOT + train_folder_add + "/model-final.h5"
	train_network(imagePath, labelPath, outputPath, modelPath = modelPath, BN_flag=3, imagePath_advp=imagePath_add, labelPath_advp=labelPath_add, pytorch_flag=pytorch_flag)

	modelPath = outputPath + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder)
	labelPath = DATASET_ROOT + labelName

	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + train_folder_add + suffix + "_add)_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=3, pathID=pathID, pytorch_flag=pytorch_flag)

def single_test_DANN():
	train_folder = "valB"
	val_folder = "valB"
	train_folder_add = "trainH"
	val_folder_add = "valH"
	#suffix = "_similarBN40"
	suffix = ""
	pathID = 0
	modelPath = ""
	pytorch_flag = True

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	imagePath_add = DATASET_ROOT + train_folder_add + "/"
	labelName_add = get_label_file_name(train_folder_add, suffix)
	labelPath_add = DATASET_ROOT + labelName_add

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + train_folder_add + suffix + "_dann/"
	#modelPath = TRAIN_OUTPUT_ROOT + train_folder_add + "/model-final.h5"
	train_network(imagePath, labelPath, outputPath, modelPath = modelPath, BN_flag=4, imagePath_advp=imagePath_add, labelPath_advp=labelPath_add, pytorch_flag=pytorch_flag)

	modelPath = outputPath + "/model-final.h5"
	if pytorch_flag:
		modelPath = outputPath + "/model-final.pth"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder)
	labelPath = DATASET_ROOT + labelName

	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + train_folder_add + suffix + "_dann)_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=4, pathID=pathID, pytorch_flag=pytorch_flag)


def test_AdvProp(subtask_id=-1):
	train_folder = "trainB"
	val_folder = "valB"
	train_folder_advp = "trainC1"
	val_folder_advp = "valC1"
	#suffix = "_similarBN1"
	suffix = ""
	test_pathID = 0

	#train_ratio_list = [0.5, 0.1, 0.01, 0.001]
	train_ratio_list = [1]
	#train_ratio_advp_list = [1]

	if subtask_id == '0':
		train_folder_advp = "trainA"
		val_folder_advp = "valA"
	elif subtask_id == '1':	
		train_folder_advp = "trainC1"
		val_folder_advp = "valC1"
	elif subtask_id == '2':	
		train_folder_advp = "trainB_blur_5"
		val_folder_advp = "valB_blur_5"
	elif subtask_id == '3':	
		train_folder_advp = "trainB_noise_5"
		val_folder_advp = "valB_noise_5"
	else:	
		return

	for i in range(len(train_ratio_list)):
		trainRatio = train_ratio_list[i]
		#train_ratio_advp = train_ratio_advp_list[i]

		imagePath = DATASET_ROOT + train_folder + "/"
		labelName = get_label_file_name(train_folder)
		labelPath = DATASET_ROOT + labelName

		imagePath_advp = DATASET_ROOT + train_folder_advp + "/"
		labelName_advp = get_label_file_name(train_folder_advp, suffix)
		labelPath_advp = DATASET_ROOT + labelName_advp

		trainOutputFolder = train_folder + "_left_" + train_folder_advp + "_" + str(trainRatio) + suffix + "_advp"
		trainOutputPath = TRAIN_OUTPUT_ROOT + trainOutputFolder + "/"
		train_network(imagePath, labelPath, trainOutputPath, trainRatio=trainRatio, BN_flag=2, imagePath_advp=imagePath_advp, labelPath_advp=labelPath_advp)

		modelPath = trainOutputPath + "/model-final.h5"

		imagePath = DATASET_ROOT + val_folder + "/"
		labelName = get_label_file_name(val_folder)
		labelPath = DATASET_ROOT + labelName

		outputPath = TEST_OUTPUT_ROOT + "(" + trainOutputFolder + ")(" + val_folder + ")/test_result.txt"
		#modelPath = ""
		test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=2, pathID=test_pathID)


def multi_factor_search_test(subtask_id=-1):
	train_folder = "trainB"
	val_folder = "valB"
	modelPath = ""

	if subtask_id == '0':
		train_folder = "trainB"
		val_folder = "valB"
	elif subtask_id == '1':	
		train_folder = "trainHm"
		val_folder = "valHm"
	elif subtask_id == '2':	
		train_folder = "trainAds"
		val_folder = "valAds"
		# modelPath = TRAIN_OUTPUT_ROOT + "trainAds_all_rob_20epoch_retrain/model-final.h5"
		# modelPath = TRAIN_OUTPUT_ROOT + "trainAds_all_rob_20epoch_retrain/model-final.h5"
		modelPath = TRAIN_OUTPUT_ROOT + "trainAds_all_rob_20epoch_single_retrain/model-final.h5"
	elif subtask_id == '3':	
		train_folder = "trainHc"
		val_folder = "valHc"
		# modelPath = TRAIN_OUTPUT_ROOT + "trainHc_all_rob_20epoch_retrain/model-final.h5"
		# modelPath = TRAIN_OUTPUT_ROOT + "trainHc_all_rob_20epoch_multi_retrain/model-final.h5"
		modelPath = TRAIN_OUTPUT_ROOT + "trainHc_all_rob_20epoch_single_retrain/model-final.h5"
	else:	
		return

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	trainOurputFolder = train_folder + "_all_rob_20epoch_single_retrain"
	# trainOurputFolder = train_folder + "_all_rob_20epoch_multi_retrain"
	trainOutputPath = TRAIN_OUTPUT_ROOT + trainOurputFolder + "/"
	#modelPath = TRAIN_OUTPUT_ROOT + "trainB_quality_channelGSY/model-final.h5"
	train_network_multi_factor_search(imagePath, labelPath, trainOutputPath, modelPath=modelPath)

	modelPath = trainOutputPath + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder, "")
	labelPath = DATASET_ROOT + labelName
	#labelPath = DATASET_ROOT + "labelsB_train.csv"

	valOutputPath = TEST_OUTPUT_ROOT + "(" + trainOurputFolder + ")_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, valOutputPath)


def unit_test_for_style():
	TRAIN_LIST = ["trainB_Maxup_pytorch"]
	TRAIN_LIST = ["trainAds_pytorch"]
	TRAIN_LIST = ["trainB"]
	TRAIN_LIST = ["trainHs"]

	TRAIN_LIST = ["trainHcs_pytorch"]
	TRAIN_LIST = ["trainHc_all_rob_50epoch_retrain"]
	TRAIN_LIST = ["trainHc_all_rob_20epoch_single_retrain"]

	TRAIN_LIST = ["trainAds_pytorch"]
	TRAIN_LIST = ["trainAds_all_rob_20epoch_retrain"]
	TRAIN_LIST = ["trainAds_all_rob_20epoch_single_retrain"]

	VAL_LIST = ["valB", \
				"valB_blur_1", "valB_blur_2", "valB_blur_3", "valB_blur_4", "valB_blur_5", \
				"valB_noise_1", "valB_noise_2", "valB_noise_3", "valB_noise_4", "valB_noise_5", \
				"valB_distort_1", "valB_distort_2", "valB_distort_3", "valB_distort_4", "valB_distort_5", \
				"valB_R_darker_1", "valB_R_darker_2", "valB_R_darker_3", "valB_R_darker_4", "valB_R_darker_5", \
				"valB_R_lighter_1", "valB_R_lighter_2", "valB_R_lighter_3", "valB_R_lighter_4", "valB_R_lighter_5", \
				"valB_G_darker_1", "valB_G_darker_2", "valB_G_darker_3", "valB_G_darker_4", "valB_G_darker_5", \
				"valB_G_lighter_1", "valB_G_lighter_2", "valB_G_lighter_3", "valB_G_lighter_4", "valB_G_lighter_5", \
				"valB_B_darker_1", "valB_B_darker_2", "valB_B_darker_3", "valB_B_darker_4", "valB_B_darker_5", \
				"valB_B_lighter_1", "valB_B_lighter_2", "valB_B_lighter_3", "valB_B_lighter_4", "valB_B_lighter_5", \
				"valB_H_darker_1", "valB_H_darker_2", "valB_H_darker_3", "valB_H_darker_4", "valB_H_darker_5", \
				"valB_H_lighter_1", "valB_H_lighter_2", "valB_H_lighter_3", "valB_H_lighter_4", "valB_H_lighter_5", \
				"valB_S_darker_1", "valB_S_darker_2", "valB_S_darker_3", "valB_S_darker_4", "valB_S_darker_5", \
				"valB_S_lighter_1", "valB_S_lighter_2", "valB_S_lighter_3", "valB_S_lighter_4", "valB_S_lighter_5", \
				"valB_V_darker_1", "valB_V_darker_2", "valB_V_darker_3", "valB_V_darker_4", "valB_V_darker_5", \
				"valB_V_lighter_1", "valB_V_lighter_2", "valB_V_lighter_3", "valB_V_lighter_4", "valB_V_lighter_5", \
				# "valB_combined_1_3", "valB_combined_1_4", "valB_combined_1_7", \
				# "valB_combined_1_8", "valB_combined_1_9", "valB_combined_1_10", \
				"valB_combined_1_0", "valB_combined_2_0", "valB_combined_3_0", \
				"valB_combined_4_0", "valB_combined_5_0", "valB_combined_6_0", \
				"valB_IMGC_motion_blur_1", "valB_IMGC_motion_blur_2", "valB_IMGC_motion_blur_3", \
				"valB_IMGC_motion_blur_4", "valB_IMGC_motion_blur_5", \
				"valB_IMGC_zoom_blur_1", "valB_IMGC_zoom_blur_2", "valB_IMGC_zoom_blur_3", \
				"valB_IMGC_zoom_blur_4", "valB_IMGC_zoom_blur_5", \
				"valB_IMGC_pixelate_1", "valB_IMGC_pixelate_2", "valB_IMGC_pixelate_3", \
				"valB_IMGC_pixelate_4", "valB_IMGC_pixelate_5", \
				"valB_IMGC_jpeg_compression_1", "valB_IMGC_jpeg_compression_2", "valB_IMGC_jpeg_compression_3", \
				"valB_IMGC_jpeg_compression_4", "valB_IMGC_jpeg_compression_5", \
				"valB_IMGC_snow_1", "valB_IMGC_snow_2", "valB_IMGC_snow_3", \
				"valB_IMGC_snow_4", "valB_IMGC_snow_5", \
				"valB_IMGC_frost_1", "valB_IMGC_frost_2", "valB_IMGC_frost_3", \
				"valB_IMGC_frost_4", "valB_IMGC_frost_5", \
				"valB_IMGC_fog_1", "valB_IMGC_fog_2", "valB_IMGC_fog_3", \
				"valB_IMGC_fog_4", "valB_IMGC_fog_5"
				]

	for i in range(len(VAL_LIST)):
		# VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valHc')
		VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valAds')

	MA_list = []
	for train_folder in TRAIN_LIST:
		imagePath = DATASET_ROOT + train_folder + "/"
		labelName = get_label_file_name(train_folder)
		labelPath = DATASET_ROOT + labelName
		outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
		if "_pytorch" in train_folder:
			pytorch_flag = True
		else:
			pytorch_flag = False

		#train_network(imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag)

		modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"
		if pytorch_flag:
			modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.pth"

		net=""
		if modelPath != "":
			if pytorch_flag:
				net = create_nvidia_network_pytorch(0, False, 0, 3)
				net.load_state_dict(torch.load(modelPath))
				net.eval()
				net.cuda()
			else:
				net = create_nvidia_network(0, False, 0, 3)
				net.load_weights(modelPath)

		for val_folder in VAL_LIST:
			# val_folder = val_folder.replace("train", "val")

			#if not (train_folder == "trainA_MUNIT_GAN_1" or val_folder == "valA_MUNIT_GAN"):
			#	continue

			imagePath = DATASET_ROOT + val_folder + "/"
			labelName = get_label_file_name(val_folder)
			labelPath = DATASET_ROOT + labelName
			outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
			MA = test_network(modelPath, imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag, net=net) #, visualize=True
			MA_list.append(MA)

		for MA in MA_list:
			print("{:.2f}\t".format(MA*100))


		MA_list = np.array(MA_list) * 100

		res_scene1 = MA_list[0]
		res_scene2 = MA_list[1:76]
		res_scene3 = MA_list[76:82]
		res_scene4 = MA_list[82:117]

		print("scene1\t{:.2f}".format(res_scene1))
		print("scene2\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene2), np.min(res_scene2), np.max(res_scene2)))
		print("scene3\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene3), np.min(res_scene3), np.max(res_scene3)))
		print("scene4\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene4), np.min(res_scene4), np.max(res_scene4)))

def unit_test_for_quality(subtask_id=-1):
	TRAIN_LIST_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5"],
					["trainB", "trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5"],
					["trainB", "trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5"]]

	
	if subtask_id == '0':
		TRAIN_LIST_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5"]]
	elif subtask_id == '1':
		TRAIN_LIST_LIST = [["trainB", "trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5"]]
	elif subtask_id == '2':
		TRAIN_LIST_LIST = [["trainB", "trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5"]]
	elif subtask_id == '3':
		TRAIN_LIST_LIST = [["trainB", "trainB_B_darker", "trainB_B_lighter", "trainB_G_darker", "trainB_G_lighter", "trainB_R_darker", "trainB_R_lighter"]]
	elif subtask_id == '4':
		TRAIN_LIST_LIST = [["trainB", "trainB_H_darker", "trainB_H_lighter", "trainB_S_darker", "trainB_S_lighter", "trainB_V_darker", "trainB_V_lighter"]]
	elif subtask_id == '5':
		TRAIN_LIST_LIST = [["trainB", "trainB_Y_luma_darker", "trainB_Y_luma_lighter", "trainB_U_blueproj_darker", "trainB_U_blueproj_lighter", "trainB_V_redproj_darker", "trainB_V_redproj_lighter"]]
	elif subtask_id == '6':
		TRAIN_LIST_LIST = [["trainB_B_darker_2", "trainB_B_lighter_2", "trainB_G_darker_2", "trainB_G_lighter_2", "trainB_R_darker_2", "trainB_R_lighter_2"]]
	elif subtask_id == '7':
		TRAIN_LIST_LIST = [["trainB_H_darker", "trainB_H_lighter", "trainB_S_darker", "trainB_S_lighter", "trainB_V_darker", "trainB_V_lighter", \
							"trainB_H_darker_2", "trainB_H_lighter_2", "trainB_S_darker_2", "trainB_S_lighter_2", "trainB_V_darker_2", "trainB_V_lighter_2", "trainB"]]
	elif subtask_id == '8':
		TRAIN_LIST_LIST = [["trainB_Y_luma_darker_2", "trainB_Y_luma_lighter_2", "trainB_U_blueproj_darker_2", "trainB_U_blueproj_lighter_2", "trainB_V_redproj_darker_2", "trainB_V_redproj_lighter_2"]]
	else:	
		return
	

	for TRAIN_LIST in TRAIN_LIST_LIST:
		for train_folder in TRAIN_LIST:
			if train_folder != "trainB":
				imagePath = DATASET_ROOT + train_folder + "/"
				labelPath = DATASET_ROOT + "labelsB_train.csv"
				outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
				train_network(imagePath, labelPath, outputPath)

			for val_folder in TRAIN_LIST:
				modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"
				#if train_folder != val_folder:
				#	continue
				val_folder = val_folder.replace("train", "val")
#				if train_folder != "trainB" and val_folder != "valB":
#					continue

				imagePath = DATASET_ROOT + val_folder + "/"
				labelPath = DATASET_ROOT + "labelsB_val.csv"
				outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
				test_network(modelPath, imagePath, labelPath, outputPath)

def combination_test_for_style(subtask_id):
	TRAIN_FOLDER_LIST = [["trainB", "trainA"],
					["trainB", "trainA_fake_GAN"],
					["trainB", "trainA_fake_color"],
					["trainB", "trainA", "trainA_fake_GAN", "trainA_fake_color"]]
	VAL_LIST = [["valB", "valB_comb"]]
	#TRAIN_RATIO_LIST = [0.25, 0.5, 0.75, 1.0]
	BN_flag = 0
	pack_in_channel = False
	pytorch_flag = False

	if subtask_id == '0':
		TRAIN_FOLDER_LIST = [["trainA"]]
		TRAIN_RATIO_LIST = [[0.25], [0.5], [0.75], [1.0]]
	elif subtask_id == '1':
		TRAIN_FOLDER_LIST = [["trainA_fake_GAN"]]
		#TRAIN_RATIO_LIST = [[0.25], [0.5], [0.75], [1.0]]
		TRAIN_RATIO_LIST = [[0.25], [0.5]]
		#TRAIN_RATIO_LIST = [[1.0]]
		#TRAIN_RATIO_LIST = [[0.5], [0.25]]
	elif subtask_id == '2':
		TRAIN_FOLDER_LIST = [["trainA_MUNIT_GAN"]]
		TRAIN_RATIO_LIST = [[0.25], [0.5], [0.75], [1.0]]
	elif subtask_id == '3':
		TRAIN_FOLDER_LIST = [["trainA", "trainA_fake_GAN", "trainA_MUNIT_GAN"]]
		TRAIN_RATIO_LIST = [[0.25,0.25,0.25], [0.5,0.5,0.5], [0.75,0.75,0.75], [1.0,1.0,1.0]]
	elif subtask_id == '4':
		TRAIN_FOLDER_LIST = [["trainA", "trainB"]]
		TRAIN_RATIO_LIST = [[0.25,1], [0.5,1], [0.75,1], [1.0,1]]
	elif subtask_id == '5':
		TRAIN_FOLDER_LIST = [["trainA_fake_GAN", "trainB"]]
		TRAIN_RATIO_LIST = [[0.25,1], [0.5,1], [0.75,1], [1.0,1]]
	elif subtask_id == '6':
		TRAIN_FOLDER_LIST = [["trainA_MUNIT_GAN", "trainB"]]
		TRAIN_RATIO_LIST = [[0.25,1], [0.5,1], [0.75,1], [1.0,1]]
	elif subtask_id == '7':
		TRAIN_FOLDER_LIST = [["trainA", "trainA_fake_GAN", "trainA_MUNIT_GAN", "trainB"]]
		TRAIN_RATIO_LIST = [[0.25,0.25,0.25,1], [0.5,0.5,0.5,1], [0.75,0.75,0.75,1], [1.0,1.0,1.0,1]]
	elif subtask_id == '8':
		TRAIN_FOLDER_LIST = [["trainA", "trainB"]]
		TRAIN_RATIO_LIST = [[1,0.5], [1,0.1], [1,0.01], [1,0.001]]
	elif subtask_id == '9':
		TRAIN_FOLDER_LIST = [["trainC1", "trainB"]]
		#TRAIN_RATIO_LIST = [[1,1], [1,0.5], [1,0.1], [1,0.01], [1,0.001]]
		TRAIN_RATIO_LIST = [[1,0.5], [1,0.1], [1,0.01], [1,0.001]]
		BN_flag = 1
	elif subtask_id == '10':
		TRAIN_FOLDER_LIST = [["trainA", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
	elif subtask_id == '11':
		TRAIN_FOLDER_LIST = [["trainB_blur_5", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
	elif subtask_id == '12':
		TRAIN_FOLDER_LIST = [["trainB_noise_5", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
	elif subtask_id == '13':
		TRAIN_FOLDER_LIST = [["trainB_distort_5", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
	elif subtask_id == '14':
		TRAIN_FOLDER_LIST = [["trainB_R_darker", "trainB_R_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '15':
		TRAIN_FOLDER_LIST = [["trainB_G_darker", "trainB_G_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '16':
		TRAIN_FOLDER_LIST = [["trainB_B_darker", "trainB_B_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '17':
		TRAIN_FOLDER_LIST = [["trainB_lap_blur", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
		pack_in_channel = True
	elif subtask_id == '18':
		TRAIN_FOLDER_LIST = [["trainB_comb", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
		pack_in_channel = True
	elif subtask_id == '19':
		TRAIN_FOLDER_LIST = [["trainB_lap_blur"]]
		VAL_LIST = ["valB_lap_blur"]
		TRAIN_RATIO_LIST = [[1]]
		BN_flag = 0
	elif subtask_id == '20':
		TRAIN_FOLDER_LIST = [["trainB_comb"]]
		VAL_LIST = ["valB_comb"]
		TRAIN_RATIO_LIST = [[1]]
		BN_flag = 0
	elif subtask_id == '21':
		TRAIN_FOLDER_LIST = [["trainN"]]
		VAL_LIST = ["valB"]
		TRAIN_RATIO_LIST = [[1]]
		BN_flag = 0
	elif subtask_id == '22':
		TRAIN_FOLDER_LIST = [["trainN", "trainB"]]
		VAL_LIST = ["valB"]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
	elif subtask_id == '21':
		TRAIN_FOLDER_LIST = [["trainN"]]
		VAL_LIST = ["valB"]
		TRAIN_RATIO_LIST = [[1]]
		BN_flag = 0
	elif subtask_id == '22':
		TRAIN_FOLDER_LIST = [["trainN", "trainB"]]
		VAL_LIST = ["valB"]
		TRAIN_RATIO_LIST = [[1,1]]
		BN_flag = 0
	elif subtask_id == '23':
		TRAIN_FOLDER_LIST = [["trainB_H_darker", "trainB_H_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '24':
		TRAIN_FOLDER_LIST = [["trainB_S_darker", "trainB_S_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '25':
		TRAIN_FOLDER_LIST = [["trainB_V_darker", "trainB_V_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '26':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5"]]
		TRAIN_RATIO_LIST = [[1,1,1,1,1,1]]
		BN_flag = 0
	elif subtask_id == '27':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5"]]
		TRAIN_RATIO_LIST = [[1,1,1,1,1,1]]
		BN_flag = 0
	elif subtask_id == '28':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5"]]
		TRAIN_RATIO_LIST = [[1,1,1,1,1,1]]
		BN_flag = 0
	elif subtask_id == '29':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5", \
		"trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5",\
		"trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5",\
		"trainB_G_darker", "trainB_G_lighter",\
		"trainB_S_darker", "trainB_S_lighter",\
		"trainB_Y_luma_darker", "trainB_Y_luma_lighter"
		]]
		TRAIN_RATIO_LIST = [1]*len(TRAIN_FOLDER_LIST[0])
		TRAIN_RATIO_LIST = [TRAIN_RATIO_LIST]
		BN_flag = 0
	elif subtask_id == '30':
		TRAIN_FOLDER_LIST = [["trainB_Y_luma_darker", "trainB_Y_luma_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '31':
		TRAIN_FOLDER_LIST = [["trainB_U_blueproj_darker", "trainB_U_blueproj_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '32':
		TRAIN_FOLDER_LIST = [["trainB_V_redproj_darker", "trainB_V_redproj_lighter", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1,1]]
		BN_flag = 0
	elif subtask_id == '33':
		TRAIN_FOLDER_LIST = [["trainB"]]
		TRAIN_RATIO_LIST = [[1]]
		BN_flag = 0
		pack_in_channel = True
	elif subtask_id == '34':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5", \
		"trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5",\
		"trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5",\
		"trainB_R_darker_2", "trainB_R_darker", "trainB_R_lighter", "trainB_R_lighter_2",\
		"trainB_G_darker_2", "trainB_G_darker", "trainB_G_lighter", "trainB_G_lighter_2",\
		"trainB_B_darker_2", "trainB_B_darker", "trainB_B_lighter", "trainB_B_lighter_2",\
		"trainB_H_darker_2", "trainB_H_darker", "trainB_H_lighter", "trainB_H_lighter_2",\
		"trainB_S_darker_2", "trainB_S_darker", "trainB_S_lighter", "trainB_S_lighter_2",\
		"trainB_V_darker_2", "trainB_V_darker", "trainB_V_lighter", "trainB_V_lighter_2"
		]]
		TRAIN_RATIO_LIST = [1]*len(TRAIN_FOLDER_LIST[0])
		TRAIN_RATIO_LIST = [TRAIN_RATIO_LIST]
	elif subtask_id == '35':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_blur_1", "trainB_blur_3", "trainB_blur_5", \
		"trainB_noise_1", "trainB_noise_3", "trainB_noise_5",\
		"trainB_distort_1", "trainB_distort_3", "trainB_distort_5",\
		"trainB_R_darker_1", "trainB_R_darker_3", "trainB_R_darker_5", \
		"trainB_R_lighter_1", "trainB_R_lighter_3", "trainB_R_lighter_5", \
		"trainB_G_darker_1", "trainB_G_darker_3", "trainB_G_darker_5", \
		"trainB_G_lighter_1", "trainB_G_lighter_3", "trainB_G_lighter_5", \
		"trainB_B_darker_1", "trainB_B_darker_3", "trainB_B_darker_5", \
		"trainB_B_lighter_1", "trainB_B_lighter_3", "trainB_B_lighter_5", \
		"trainB_H_darker_1", "trainB_H_darker_3", "trainB_H_darker_5", \
		"trainB_H_lighter_1", "trainB_H_lighter_3", "trainB_H_lighter_5", \
		"trainB_S_darker_1", "trainB_S_darker_3", "trainB_S_darker_5", \
		"trainB_S_lighter_1", "trainB_S_lighter_3", "trainB_S_lighter_5", \
		"trainB_V_darker_1", "trainB_V_darker_3", "trainB_V_darker_5", \
		"trainB_V_lighter_1", "trainB_V_lighter_3", "trainB_V_lighter_5"
		]]
		TRAIN_RATIO_LIST = [1]*len(TRAIN_FOLDER_LIST[0])
		TRAIN_RATIO_LIST = [TRAIN_RATIO_LIST]
	elif subtask_id == '36':
		TRAIN_FOLDER_LIST = [["trainC1", "trainB"]]
		TRAIN_RATIO_LIST = [[1,1]]
	elif subtask_id == '37':
		TRAIN_FOLDER_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5", \
		"trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5",\
		#"trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5",\
		"trainB_R_darker_1", "trainB_R_darker_2", "trainB_R_darker_3", "trainB_R_darker_4", "trainB_R_darker_5", \
		"trainB_R_lighter_1", "trainB_R_lighter_2", "trainB_R_lighter_3", "trainB_R_lighter_4", "trainB_R_lighter_5", \
		"trainB_G_darker_1", "trainB_G_darker_2", "trainB_G_darker_3", "trainB_G_darker_4", "trainB_G_darker_5", \
		"trainB_G_lighter_1", "trainB_G_lighter_2", "trainB_G_lighter_3", "trainB_G_lighter_4", "trainB_G_lighter_5", \
		"trainB_B_darker_1", "trainB_B_darker_2", "trainB_B_darker_3", "trainB_B_darker_4", "trainB_B_darker_5", \
		"trainB_B_lighter_1", "trainB_B_lighter_2", "trainB_B_lighter_3", "trainB_B_lighter_4", "trainB_B_lighter_5", \
		"trainB_H_darker_1", "trainB_H_darker_2", "trainB_H_darker_3", "trainB_H_darker_4", "trainB_H_darker_5", \
		"trainB_H_lighter_1", "trainB_H_lighter_2", "trainB_H_lighter_3", "trainB_H_lighter_4", "trainB_H_lighter_5", \
		"trainB_S_darker_1", "trainB_S_darker_2", "trainB_S_darker_3", "trainB_S_darker_4", "trainB_S_darker_5", \
		"trainB_S_lighter_1", "trainB_S_lighter_2", "trainB_S_lighter_3", "trainB_S_lighter_4", "trainB_S_lighter_5", \
		"trainB_V_darker_1", "trainB_V_darker_2", "trainB_V_darker_3", "trainB_V_darker_4", "trainB_V_darker_5", \
		"trainB_V_lighter_1", "trainB_V_lighter_2", "trainB_V_lighter_3", "trainB_V_lighter_4", "trainB_V_lighter_5"
		]]
		TRAIN_RATIO_LIST = [1]*len(TRAIN_FOLDER_LIST[0])
		TRAIN_RATIO_LIST = [TRAIN_RATIO_LIST]
		pytorch_flag = True
	elif subtask_id == '100':
		#For test
		TRAIN_FOLDER_LIST = [["trainB_blur"]]
		TRAIN_RATIO_LIST = [[1]]
		VAL_LIST = [["valB", "valB_blur_1", "valB_blur_2", "valB_blur_3", "valB_blur_4", "valB_blur_5"]]
	else:
		print('invalid subtask_id!!!')
		return

	#pretrain_model = "tmp"
	#pretrain_model_path = TRAIN_OUTPUT_ROOT + pretrain_model + "/model-final.h5"
	pretrain_model_path = ""
	suffix = ""
	if BN_flag > 0:
		suffix = suffix + "_BN" + str(BN_flag)
	if pack_in_channel:
		suffix = suffix + "_pack_channel"
	if pytorch_flag:
		suffix = suffix + "_pytorch"

	i = 0
	for train_folder_list in TRAIN_FOLDER_LIST:
		for train_ratio in TRAIN_RATIO_LIST:

			trainOurputFolder = "combine_" + str(subtask_id) + suffix
			#trainOurputFolder = "combine_" + str(train_folder_list) + "_" + str(train_ratio) + suffix

			trainOutputPath = TRAIN_OUTPUT_ROOT + trainOurputFolder + "/"

			imagePath_list = []
			labelPath_list = []
			for train_folder in train_folder_list:
				imagePath = DATASET_ROOT + train_folder + "/"
				labelName = get_label_file_name(train_folder)
				labelPath = DATASET_ROOT + labelName
				imagePath_list.append(imagePath)
				labelPath_list.append(labelPath)
			train_network_multi(imagePath_list, labelPath_list, trainOutputPath, pretrain_model_path, BN_flag=BN_flag, trainRatio=train_ratio, pack_flag=pack_in_channel, pytorch_flag=pytorch_flag)

			modelPath = trainOutputPath + "model-final.h5"
			for val_folder_list in VAL_LIST:
				imagePath_list = []
				labelPath_list = []
				for val_folder in val_folder_list:
					imagePath = DATASET_ROOT + val_folder + "/"
					labelName = get_label_file_name(val_folder)
					labelPath = DATASET_ROOT + labelName
					imagePath_list.append(imagePath)
					labelPath_list.append(labelPath)

				valOutputPath = TEST_OUTPUT_ROOT + "(" + trainOurputFolder + ")_(" + str(val_folder_list) + ")/test_result.txt"
				test_network_multi(modelPath, imagePath_list, labelPath_list, valOutputPath, BN_flag=BN_flag, pack_flag=pack_in_channel, pytorch_flag=pytorch_flag)

		i += 1

def combination_test_for_style_pretrain(subtask_id=-1):
	#TRAIN_RATIO_LIST = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
	#TRAIN_RATIO_LIST = [0.25, 0.5, 0.75, 1]
	
	#TRAIN_RATIO_LIST = [1]
	TRAIN_RATIO_LIST = [1]
	
	#PRETRAIN_MODEL_LIST = ["trainA"]
	#PRETRAIN_MODEL_LIST = ["combine0"]
	#PRETRAIN_MODEL_LIST = ["trainA_fake_GAN_1", "trainA_fake_GAN_0.75", "trainA_fake_GAN_0.5", "trainA_fake_GAN_0.25"]
	#PRETRAIN_MODEL_LIST = ["trainC1_BN0_classify"]
	#PRETRAIN_MODEL_LIST = ["trainC1"]
	PRETRAIN_MODEL_LIST = [""]
	#TRAIN_LIST = ["trainB"]
	TRAIN_LIST = ["trainB"]
	#VAL_LIST = ["valB", "valA"]
	VAL_LIST = ["valB"]
	partial_preModel = False
	reinit_header = False
	reinit_BN = False
	BN_flag = 0
	classification = False

	
	if subtask_id == '0':
		PRETRAIN_MODEL_LIST = ["trainA_0.25", "trainA_0.5", "trainA_0.75", "trainA_1"]
	elif subtask_id == '1':
		PRETRAIN_MODEL_LIST = ["trainA_fake_GAN_0.25", "trainA_fake_GAN_0.5", "trainA_fake_GAN_0.75", "trainA_fake_GAN_1"]
	elif subtask_id == '2':
		PRETRAIN_MODEL_LIST = ["trainA_MUNIT_GAN_0.25", "trainA_MUNIT_GAN_0.5", "trainA_MUNIT_GAN_0.75", "trainA_MUNIT_GAN_1"]
	elif subtask_id == '3':
		PRETRAIN_MODEL_LIST = ["trainA_ALL_0.25", "trainA_ALL_0.5", "trainA_ALL_0.75", "trainA_ALL_1"]
		#PRETRAIN_MODEL_LIST = ["combine_0_3_[0.25, 0.25, 0.25]", "combine_0_3_[1.0, 1.0, 1.0]"]
	elif subtask_id == '4':
		PRETRAIN_MODEL_LIST = ["trainA", "trainA_fake_GAN", "trainA_MUNIT_GAN"]
	elif subtask_id == '5':
		PRETRAIN_MODEL_LIST = ["trainA_BN", "trainC1_BN"]
		#PRETRAIN_MODEL_LIST = ["trainC1_BN1_classify"]
	elif subtask_id == '6':
		PRETRAIN_MODEL_LIST = ["trainC1"]
	elif subtask_id == '7':
		PRETRAIN_MODEL_LIST = ["trainC1_BN"]
	elif subtask_id == '8':
		PRETRAIN_MODEL_LIST = ["trainB"]
		BN_flag = 0
	elif subtask_id == '9':
		PRETRAIN_MODEL_LIST = ["trainB_BN1_2000"]
		BN_flag = 1
	#else:
	#	return
	

	#suffix = "_pretrain"
	suffix = "_2000"
	if partial_preModel:
		suffix = "_partialpretrain"
	if reinit_header:
		suffix = suffix + "_reinitheader"
	if BN_flag > 0:
		suffix = suffix + "_BN" + str(BN_flag)
	if reinit_BN:
		suffix = suffix + "_reinitBN"
	if classification:
		suffix = suffix + "_classify"

	id = 0
	for train_ratio in TRAIN_RATIO_LIST:
		for pretrain_model in PRETRAIN_MODEL_LIST:
			for train_folder in TRAIN_LIST:
				id += 1
				print(id)
				imagePath = DATASET_ROOT + train_folder + "/"
				labelName = get_label_file_name(train_folder)
				labelPath = DATASET_ROOT + labelName

				if pretrain_model == "":
					trainOutputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + str(train_ratio) + "/"
					pretrain_model_path = ""
				else:

					trainOutputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + str(train_ratio) + "_(" + pretrain_model + suffix + ")/"

					pretrain_model_path = TRAIN_OUTPUT_ROOT + pretrain_model + "/model-final.h5"

				#if (train_ratio == 1 and pretrain_model == ""):
				#	continue

				train_network(imagePath, labelPath, trainOutputPath, pretrain_model_path, trainRatio=train_ratio, partialPreModel=partial_preModel, reinitHeader=reinit_header, BN_flag=BN_flag, reinitBN=reinit_BN, classification=classification)

				for val_folder in VAL_LIST:
					modelPath = trainOutputPath + "/model-final.h5"
					val_folder = val_folder.replace("train", "val")

					imagePath = DATASET_ROOT + val_folder + "/"
					labelName = get_label_file_name(val_folder)
					labelPath = DATASET_ROOT + labelName

					outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + "_" + str(train_ratio) + "_(" + pretrain_model + suffix + ")" + ")_(" + val_folder + ")/test_result.txt"
					test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=BN_flag, classification=classification)

def calculate_FID_folder(subtask_id=-1):
	datasets = ["trainB_small", "trainA_small"]
	outputFile = "small_fid.txt"

	if subtask_id == '0':
		datasets = ["trainB", "trainA", "trainB_fake_GAN", "trainA_fake_GAN", "trainB_fake_color", "trainA_fake_color", "trainC1"]
		outputFile = "style_fid.txt"
	elif subtask_id == '1':
		datasets = ["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5"]
		outputFile = "blur_fid.txt"
	elif subtask_id == '2':
		datasets = ["trainB", "trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5"]
		outputFile = "noise_fid.txt"
	elif subtask_id == '3':
		datasets = ["trainB", "trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5"]
		outputFile = "distortion_fid.txt"
	elif subtask_id == '4':
		datasets = ["valB", "valB_R_darker", "valB_R_lighter", "valB_R_darker_2", "valB_R_lighter_2"]
		outputFile = "R_fid.txt"
	elif subtask_id == '5':
		datasets = ["valB", "valB_G_darker", "valB_G_lighter", "valB_G_darker_2", "valB_G_lighter_2"]
		outputFile = "G_fid.txt"
	elif subtask_id == '6':
		datasets = ["valB", "valB_B_darker", "valB_B_lighter", "valB_B_darker_2", "valB_B_lighter_2"]
		outputFile = "B_fid.txt"
	elif subtask_id == '7':
		datasets = ["valB", "valB_H_darker", "valB_H_lighter", "valB_H_darker_2", "valB_H_lighter_2"]
		outputFile = "H_fid.txt"
	elif subtask_id == '8':
		datasets = ["valB", "valB_S_darker", "valB_S_lighter", "valB_S_darker_2", "valB_S_lighter_2"]
		outputFile = "S_fid.txt"
	elif subtask_id == '9':
		datasets = ["valB", "valB_V_darker", "valB_V_lighter", "valB_V_darker_2", "valB_V_lighter_2"]
		outputFile = "V_fid.txt"
	elif subtask_id == '10':
		datasets = ["valB", "valB_R_darker_1", "valB_R_darker_2", "valB_R_darker_3", "valB_R_darker_4", \
							"valB_R_lighter_1", "valB_R_lighter_2", "valB_R_lighter_3", "valB_R_lighter_4", \
							"valB_G_darker_1", "valB_G_darker_2", "valB_G_darker_3", "valB_G_darker_4", \
							"valB_G_lighter_1", "valB_G_lighter_2", "valB_G_lighter_3", "valB_G_lighter_4", \
							"valB_B_darker_1", "valB_B_darker_2", "valB_B_darker_3", "valB_B_darker_4", \
							"valB_B_lighter_1", "valB_B_lighter_2", "valB_B_lighter_3", "valB_B_lighter_4", \
							"valB_H_darker_1", "valB_H_darker_2", "valB_H_darker_3", "valB_H_darker_4", \
							"valB_H_lighter_1", "valB_H_lighter_2", "valB_H_lighter_3", "valB_H_lighter_4", \
							"valB_S_darker_1", "valB_S_darker_2", "valB_S_darker_3", "valB_S_darker_4", \
							"valB_S_lighter_1", "valB_S_lighter_2", "valB_S_lighter_3", "valB_S_lighter_4", \
							"valB_V_darker_1", "valB_V_darker_2", "valB_V_darker_3", "valB_V_darker_4", \
							"valB_V_lighter_1", "valB_V_lighter_2", "valB_V_lighter_3", "valB_V_lighter_4", \
					]
		outputFile = "RGB_HSV_fid.txt"
	elif subtask_id == '11':
		datasets = ["valB", "valB_R_darker_5", \
							"valB_R_lighter_5", \
							"valB_G_darker_5", \
							"valB_G_lighter_5", \
							"valB_B_darker_5", \
							"valB_B_lighter_5", \
							"valB_H_darker_5", \
							"valB_H_lighter_5", \
							"valB_S_darker_5", \
							"valB_S_lighter_5", \
							"valB_V_darker_5", \
							"valB_V_lighter_5", \
					]
		outputFile = "RGB_HSV_fid_065.txt"



	paths = []
	for dataset in datasets:
		paths.append(DATASET_ROOT + dataset)
	fid_value_array = calculate_fid_given_multi_paths(paths, batch_size=128, cuda=True, dims=2048)

	f = open(TEST_OUTPUT_ROOT + outputFile, 'w')
	print(TEST_OUTPUT_ROOT + outputFile)
	n_dataset = len(datasets)
	for i in range(n_dataset):
		one_line = ""
		for j in range(n_dataset):
			one_line = one_line + " & " + "{:.2f}".format(fid_value_array[i*n_dataset + j])
		one_line = one_line + '\n'
		f.write(one_line)
		f.flush()
	f.close()


def calculate_FID_list_files_succ_fail(subtask_id=-1):
	#datasets = ["(trainB)_(valB)", "(trainA)_(valA)"]
	#thresh_list = ["0.1", "0.2", "0.5", "1", "2", "5"]
	datasets = ["(trainB)_(valB)"]
	thresh_list = ["5"]
	outputFile = "succ_fail_fid.txt"

	if subtask_id == '0':
		datasets = ["(trainB)_(valB)", "(trainA)_(valA)", "(trainC1)_(valC1)"]
		thresh_list = ["0.1", "0.2", "0.5", "1", "2", "5"]
	elif subtask_id == '1':
		datasets = ["(trainB)_(trainB)", "(trainA)_(trainA)", "(trainC1)_(trainC1)"]
		thresh_list = ["0.1", "0.2", "0.5", "1", "2", "5"]
		outputFile = "succ_fail_fid_train.txt"

	f = open(TEST_OUTPUT_ROOT + outputFile, 'w')
	for dataset in datasets:
		for thresh in thresh_list:
			files = []
			files.append(OUTPUT_ROOT + "/test_results/" + dataset + "/img_list_" + str(thresh) + "_succ.txt")
			files.append(OUTPUT_ROOT + "/test_results/" + dataset + "/img_list_" + str(thresh) + "_fail.txt")
			fid_value = calculate_fid_given_list_files(files, batch_size=128, cuda=True, dims=2048)
			one_line = dataset + " " + str(thresh) + " succ-fail " + str(fid_value) + "\n"
			f.write(one_line)
			#f.flush()
	f.close()


def calculate_FID_list_files(subtask_id=-1):
	#datasets_pair = [["trainB", "trainC1"],["valB", "valC1"]]
	datasets_pair = [["trainB", "trainC1"],["valB", "valC1"]]
	#outputFile = "FID_trainB_C1_valB_C1.txt"
	#outputFile = "FID_trainB_valB_C1.txt"
	outputFile = "FID_test.txt"

	f = open(TEST_OUTPUT_ROOT + outputFile, 'w')
	files_pair = []
	folders_pair = []
	for datasets in datasets_pair:
		files = []
		folders = []
		for dataset in datasets:
			files.append(DATASET_ROOT + get_label_file_name(dataset))
			folders.append(DATASET_ROOT + dataset + '/')
		files_pair.append(files)
		folders_pair.append(folders)

	fid_value = calculate_fid_given_list_files_csv(files_pair, folders_pair, batch_size=128, cuda=True, dims=2048)
	one_line = str(datasets_pair) + " " + str(fid_value) + "\n"
	print(one_line)
	f.write(one_line)

	f.close()


def calculate_L2D():
	base_folders = ["valB"]
	
	comparison_folders =  [	"valB_blur_1", "valB_blur_2", "valB_blur_3", "valB_blur_4", "valB_blur_5", \
							"valB_noise_1", "valB_noise_2", "valB_noise_3", "valB_noise_4", "valB_noise_5", \
							"valB_distort_1", "valB_distort_2", "valB_distort_3", "valB_distort_4", "valB_distort_5", \
							"valB_R_darker_1", "valB_R_darker_2", "valB_R_darker_3", "valB_R_darker_4", "valB_R_darker_5", \
							"valB_R_lighter_1", "valB_R_lighter_2", "valB_R_lighter_3", "valB_R_lighter_4", "valB_R_lighter_5", \
							"valB_G_darker_1", "valB_G_darker_2", "valB_G_darker_3", "valB_G_darker_4", "valB_G_darker_5", \
							"valB_G_lighter_1", "valB_G_lighter_2", "valB_G_lighter_3", "valB_G_lighter_4", "valB_G_lighter_5", \
							"valB_B_darker_1", "valB_B_darker_2", "valB_B_darker_3", "valB_B_darker_4", "valB_B_darker_5", \
							"valB_B_lighter_1", "valB_B_lighter_2", "valB_B_lighter_3", "valB_B_lighter_4", "valB_B_lighter_5", \
							"valB_H_darker_1", "valB_H_darker_2", "valB_H_darker_3", "valB_H_darker_4", "valB_H_darker_5", \
							"valB_H_lighter_1", "valB_H_lighter_2", "valB_H_lighter_3", "valB_H_lighter_4", "valB_H_lighter_5", \
							"valB_S_darker_1", "valB_S_darker_2", "valB_S_darker_3", "valB_S_darker_4", "valB_S_darker_5", \
							"valB_S_lighter_1", "valB_S_lighter_2", "valB_S_lighter_3", "valB_S_lighter_4", "valB_S_lighter_5", \
							"valB_V_darker_1", "valB_V_darker_2", "valB_V_darker_3", "valB_V_darker_4", "valB_V_darker_5", \
							"valB_V_lighter_1", "valB_V_lighter_2", "valB_V_lighter_3", "valB_V_lighter_4", "valB_V_lighter_5", \
					]
	
	#comparison_folders =  [	"valB_R_lighter_1", "valB_R_lighter_2", "valB_R_lighter_3", "valB_R_lighter_4", "valB_R_lighter_5" ]

	for folderA in base_folders:
		for folderB in comparison_folders:
			#print(glob.glob(os.path.join(DATASET_ROOT, folderA, "*.jpg")))
			dist_list = []
			for image_path in glob.glob(os.path.join(DATASET_ROOT, folderA, "*.jpg")):
				img1 = np.array(cv2.imread(image_path)).astype(float)
				image_path2 = image_path.replace(folderA, folderB)
				img2 = np.array(cv2.imread(image_path2)).astype(float)
				dist = np.linalg.norm(img1-img2)
				dist_list.append(dist)

			print("{:.2f}\t".format(np.mean(dist_list)))


def visualize_model_on_image():
	radius = 5

	imageFolder = "trainB"
	imageName = "0.jpg"
	imagePath = DATASET_ROOT + imageFolder + "/" + imageName
	label = 0.0

	#imageName = "5244.jpg"
	#imagePath = DATASET_ROOT + imageFolder + "/" + imageName
	#label = -1.29
	#imagePath = DATASET_ROOT + "trainB/5244.jpg"
	#label = -1.29

	imageFolder = "valB"
	imageName = "21240.jpg"
	imagePath = DATASET_ROOT + imageFolder + "/" + imageName
	label = -7.827

	'''
	imageFolder = "valB"
	imageName = "23406.jpg"
	imagePath = DATASET_ROOT + imageFolder + "/" + imageName
	label = 5.3167

	imageFolder = "valB"
	imageName = "27054.jpg"
	imagePath = DATASET_ROOT + imageFolder + "/" + imageName
	label = 13.666
	'''

	modelFolder = "trainB"
	modelPath = TRAIN_OUTPUT_ROOT + modelFolder + "/model-final.h5"

	outputPath = TEST_OUTPUT_ROOT + "visulize_(" + modelFolder + ")(" + imageFolder + "_" + imageName + ")/heatmap" + str(radius) + ".jpg"

	visualize_network_on_image(modelPath, imagePath, label, outputPath, radius=radius)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='batch train test')
	parser.add_argument('--gpu_id', required=False, metavar="gpu_id", help='gpu id (0/1)')
	parser.add_argument('--task_id', required=False, metavar="task_id", help='task_id id (0/1)')
	parser.add_argument('--subtask_id', required=False, metavar="subtask_id", help='subtask_id id (0/1)')
	args = parser.parse_args()

	if (args.gpu_id != None):
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
	print("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])

	# from torch.multiprocessing import Pool, Process, set_start_method
	# try:
	# 	set_start_method('spawn')
	# except RuntimeError:
	# 	pass

	if args.task_id:
		if args.task_id == '0':
			single_test()
		elif args.task_id == '1':
			single_test_with_config(args.subtask_id)
		elif args.task_id == '2':
			unit_test_for_style()
		elif args.task_id == '3':
			unit_test_for_quality(args.subtask_id)
		elif args.task_id == '4':
			combination_test_for_style(args.subtask_id)
		elif args.task_id == '5':
			combination_test_for_style_pretrain(args.subtask_id)
		elif args.task_id == '6':
			test_AdvProp(args.subtask_id)
		elif args.task_id == '7':
			calculate_FID(args.subtask_id)
		elif args.task_id == '8':
			multi_factor_search_test(args.subtask_id)
		elif args.task_id == '9':
			calculate_FID_folder(args.subtask_id)
		elif args.task_id == '10':
			single_test_ImgAndFeature()
		elif args.task_id == '11':
			single_test_DANN()
		else:
			print("Unknown task: " + args.task_id)
	else:
		#single_test()
		#single_test_with_config('15')
		#single_test_AdvProp()
		#single_test_ImgAndFeature()
		#single_test_DANN()
		#multi_factor_search_test()
		unit_test_for_style()
		#unit_test_for_quality()
		#combination_test_for_style('36')
		#combination_test_for_style_pretrain()
		#test_AdvProp()
		#calculate_FID_folder('11')
		#calculate_FID_list_files_succ_fail('1')
		#calculate_FID_list_files()
		#visualize_model_on_image()
		#calculate_L2D()