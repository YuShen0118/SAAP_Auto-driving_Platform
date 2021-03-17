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
	train_folder = "trainAudi4_pytorch"
	val_folder = "valAudi4"
	pytorch_flag = True

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	# labelName = get_label_file_name("trainAudi3")
	labelPath = DATASET_ROOT + labelName

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
	# train_network(imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag)

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
	lr = 0.0001

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
	elif subtask_id == '18':
		train_folder = "trainB"
		val_folder = "valB"
		pytorch_flag = True
		BN_flag = 5 # commaai net
	elif subtask_id == '19':
		train_folder = "trainHc"
		val_folder = "valHc"
		pytorch_flag = True
		BN_flag = 5 # commaai net
	elif subtask_id == '20':
		train_folder = "trainAds"
		val_folder = "valAds"
		pytorch_flag = True
		BN_flag = 5 # commaai net
	elif subtask_id == '21':
		train_folder = "trainB"
		val_folder = "valB"
		pytorch_flag = True
		lr = 0.01
		suffix = "_lr1e_2"
	elif subtask_id == '22':
		train_folder = "trainB"
		val_folder = "valB"
		pytorch_flag = True
		lr = 0.001
		suffix = "_lr1e_3"
	elif subtask_id == '23':
		train_folder = "trainB"
		val_folder = "valB"
		pytorch_flag = True
		lr = 0.0001
		suffix = "_lr1e_4"
	elif subtask_id == '24':
		train_folder = "trainBs"
		val_folder = "valBs"
		pytorch_flag = True
		suffix = "_newp1"
		# modelPath = TRAIN_OUTPUT_ROOT + "trainB_pytorch_lr1e_4/model-final.pth"
	elif subtask_id == '25':
		train_folder = "trainHcs"
		val_folder = "valHcs"
		pytorch_flag = True
		suffix = "_newp1"
	elif subtask_id == '26':
		train_folder = "trainAds"
		val_folder = "valAds"
		pytorch_flag = True
		suffix = "_newp1"
	elif subtask_id == '27':
		train_folder = "trainBs"
		val_folder = "valBs"
		pytorch_flag = True
		suffix = "_newp_reHc1"
		modelPath = TRAIN_OUTPUT_ROOT + "trainHcs_pytorch_newp1/model-final.pth"
	elif subtask_id == '28':
		train_folder = "trainBs"
		val_folder = "valBs"
		pytorch_flag = True
		suffix = "_newp_reAds1"
		modelPath = TRAIN_OUTPUT_ROOT + "trainAds_pytorch_newp1/model-final.pth"
	elif subtask_id == '29':
		train_folder = "trainHcs"
		val_folder = "valHcs"
		pytorch_flag = True
		suffix = "_newp_reB1"
		modelPath = TRAIN_OUTPUT_ROOT + "trainB_pytorch_newp1/model-final.pth"
	elif subtask_id == '30':
		train_folder = "trainHcs"
		val_folder = "valHcs"
		pytorch_flag = True
		suffix = "_newp_reAds1"
		modelPath = TRAIN_OUTPUT_ROOT + "trainAds_pytorch_newp1/model-final.pth"
	elif subtask_id == '31':
		train_folder = "trainAds"
		val_folder = "valAds"
		pytorch_flag = True
		suffix = "_newp_reB1"
		modelPath = TRAIN_OUTPUT_ROOT + "trainB_pytorch_newp1/model-final.pth"
	elif subtask_id == '32':
		train_folder = "trainAds"
		val_folder = "valAds"
		pytorch_flag = True
		suffix = "_newp_reHcs1"
		modelPath = TRAIN_OUTPUT_ROOT + "trainHcs_pytorch_newp1/model-final.pth"
	elif subtask_id == '33':
		train_folder = "trainAudi1"
		val_folder = "valAudi1"
		pytorch_flag = True
		suffix = "_slabel"
	elif subtask_id == '34':
		train_folder = "trainAudi2"
		val_folder = "valAudi2"
		suffix = "_slabel"
		pytorch_flag = True
	elif subtask_id == '35':
		train_folder = "trainAudi1segall"
		val_folder = "valAudi1segall"
		suffix = "_slabel"
		pytorch_flag = True
	elif subtask_id == '36':
		train_folder = "trainAudi2segall"
		val_folder = "valAudi2segall"
		suffix = "_slabel"
		pytorch_flag = True
	elif subtask_id == '37':
		train_folder = "trainAudi1seg1"
		val_folder = "valAudi1seg1"
		suffix = "_slabel"
		pytorch_flag = True
	elif subtask_id == '38':
		train_folder = "trainAudi2seg1"
		val_folder = "valAudi2seg1"
		suffix = "_slabel"
		pytorch_flag = True
	elif subtask_id == '39':
		train_folder = "trainAudi1"
		val_folder = "valAudi1"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi2_pytorch_slabel/model-final.pth"
	elif subtask_id == '40':
		train_folder = "trainAudi2"
		val_folder = "valAudi2"
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi1_pytorch_slabel/model-final.pth"
		pytorch_flag = True
	elif subtask_id == '41':
		train_folder = "trainAudi1segall"
		val_folder = "valAudi1segall"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi2segall_pytorch/model-final.pth"
	elif subtask_id == '42':
		train_folder = "trainAudi2segall"
		val_folder = "valAudi2segall"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi1segall_pytorch/model-final.pth"
	elif subtask_id == '43':
		train_folder = "trainAudi1seg1"
		val_folder = "valAudi1seg1"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi2seg1_pytorch/model-final.pth"
	elif subtask_id == '44':
		train_folder = "trainAudi2seg1"
		val_folder = "valAudi2seg1"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi1seg1_pytorch/model-final.pth"
	elif subtask_id == '45':
		train_folder = "trainAudi3"
		val_folder = "valAudi3"
		pytorch_flag = True
	elif subtask_id == '46':
		train_folder = "trainAudi4"
		val_folder = "valAudi4"
		pytorch_flag = True
	elif subtask_id == '47':
		train_folder = "trainAudi3"
		val_folder = "valAudi3"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi4_pytorch/model-final.pth"
	elif subtask_id == '48':
		train_folder = "trainAudi4"
		val_folder = "valAudi4"
		pytorch_flag = True
		modelPath = TRAIN_OUTPUT_ROOT + "trainAudi3_pytorch/model-final.pth"
	elif subtask_id == '100':
		train_folder = "trainAudi5"
		val_folder = "valAudi5"
		pytorch_flag = True
		BN_flag = 8
		suffix = "_test"
	else:
		print('Invalid subtask_id ', subtask_id)
		return


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
	train_network(imagePath, labelPath, outputPath, modelPath=modelPath, BN_flag=BN_flag, classification=classification, netType=net_type, Maxup_flag=Maxup_flag, pytorch_flag=pytorch_flag, lr=lr)

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

def single_test_2_streams(subtask_id=-1):
	train_folder = "trainAudi3"
	val_folder = "valAudi3"
	train_folder_add = "trainAudi4"
	val_folder_add = "valAudi4"
	#suffix = "_similarBN40"
	labelsuffix = ""
	suffix = labelsuffix
	pathID = 0
	modelPath = ""
	pytorch_flag = True
	BN_flag = 6

	if subtask_id == '0':
		BN_flag = 6 # ADDA
		suffix = suffix + "_10t"
	elif subtask_id == '1':	
		BN_flag = 4 # DANN

	if BN_flag == 4:
		suffix = suffix + "_dann"
	elif BN_flag == 6:
		suffix = suffix + "_adda"

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	imagePath_add = DATASET_ROOT + train_folder_add + "/"
	labelName_add = get_label_file_name(train_folder_add, labelsuffix)
	labelPath_add = DATASET_ROOT + labelName_add


	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + train_folder_add + suffix + "/"
	# modelPath = TRAIN_OUTPUT_ROOT + train_folder_add + "/model-final.h5"
	modelPath = TRAIN_OUTPUT_ROOT + "trainAudi3_pytorch/model-final.pth"
	train_network(imagePath, labelPath, outputPath, modelPath = modelPath, BN_flag=BN_flag, imagePath_advp=imagePath_add, labelPath_advp=labelPath_add, pytorch_flag=pytorch_flag)

	modelPath = outputPath + "/model-final.h5"
	if pytorch_flag:
		modelPath = outputPath + "/model-final.pth"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder)
	labelPath = DATASET_ROOT + labelName

	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + train_folder_add + suffix + ")_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=BN_flag, pathID=pathID, pytorch_flag=pytorch_flag)


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
	BN_flag = 0
	train_suffix = ""
	pytorch_flag = False

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
	elif subtask_id == '4':	
		train_folder = "trainB"
		val_folder = "valB"
		BN_flag = 5
		pytorch_flag = True
	elif subtask_id == '5':	
		train_folder = "trainHc"
		val_folder = "valHc"
		BN_flag = 5
		pytorch_flag = True
	elif subtask_id == '6':	
		train_folder = "trainAds"
		val_folder = "valAds"
		BN_flag = 5
		pytorch_flag = True
	else:	
		return

	if BN_flag > 0:
		train_suffix = train_suffix + "_BN" + str(BN_flag)
	if pytorch_flag:
		train_suffix = train_suffix + "_pytorch"

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName

	trainOurputFolder = train_folder + "_all_rob_20epoch_single" + train_suffix
	# trainOurputFolder = train_folder + "_all_rob_20epoch_multi_retrain"
	trainOutputPath = TRAIN_OUTPUT_ROOT + trainOurputFolder + "/"
	#modelPath = TRAIN_OUTPUT_ROOT + "trainB_quality_channelGSY/model-final.h5"
	train_network_multi_factor_search(imagePath, labelPath, trainOutputPath, modelPath=modelPath, BN_flag=BN_flag, pytorch_flag=pytorch_flag)

	modelPath = trainOutputPath + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder, "")
	labelPath = DATASET_ROOT + labelName
	#labelPath = DATASET_ROOT + "labelsB_train.csv"

	valOutputPath = TEST_OUTPUT_ROOT + "(" + trainOurputFolder + ")_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, valOutputPath, BN_flag=BN_flag, pytorch_flag=pytorch_flag)

def reformal_corruption_accs(MA_list):
	corruption_accs_1 = [MA_list[0]]
	corruption_accs_2 = [MA_list[1:6], MA_list[6:11], MA_list[11:16], MA_list[16:26], MA_list[26:36], MA_list[36:46], MA_list[46:56], MA_list[56:66], MA_list[66:76]]
	corruption_accs_3 = [MA_list[76:77], MA_list[77:78], MA_list[78:79], MA_list[79:80], MA_list[80:81], MA_list[81:82]]
	corruption_accs_4 = [MA_list[82:87], MA_list[87:92], MA_list[92:97], MA_list[97:102], MA_list[102:107], MA_list[107:112], MA_list[112:117]]
	return corruption_accs_1, corruption_accs_2, corruption_accs_3, corruption_accs_4

def compute_mce(corruption_accs, base_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  n = len(corruption_accs)
  for i in range(n):
    avg_err = 1 - np.mean(corruption_accs[i])
    base_err = 1 - np.mean(base_accs[i])
    ce = 100 * avg_err / base_err
    mce += ce / n
  return mce

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

	TRAIN_LIST = ["trainAds_all_rob_20epoch_single_commaai_pytorch"]
	# TRAIN_LIST = ["trainHc_augmix_pytorch"]
	# TRAIN_LIST = ["trainAds_augmix_pytorch"]

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
	# VAL_LIST = ["valB"]

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
		BN_flag = 0
		if "commaai" in train_folder:
			BN_flag = 5
		if modelPath != "":
			if pytorch_flag:
				net = create_nvidia_network_pytorch(BN_flag)
				if "augmix" in train_folder:
					net = torch.nn.DataParallel(net).cuda()
					net.load_state_dict(torch.load(modelPath)['state_dict'])
				else:
					net.load_state_dict(torch.load(modelPath))
					net.cuda()
				net.eval()
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

		MA_list = np.array(MA_list) * 100

		for MA in MA_list:
			print("{:.2f}\t".format(MA))

		res_scene1 = MA_list[0]
		res_scene2 = MA_list[1:76]
		res_scene3 = MA_list[76:82]
		res_scene4 = MA_list[82:117]

		print("scene1\t{:.2f}".format(res_scene1))
		print("scene2\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene2), np.min(res_scene2), np.max(res_scene2)))
		print("scene3\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene3), np.min(res_scene3), np.max(res_scene3)))
		print("scene4\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene4), np.min(res_scene4), np.max(res_scene4)))

		# MA_list = [89.29, 89.2, 89.46, 88.75, 82.36, 75.45, 89.08, 88.66, 88.54, 85.54, 82.74, 89.14, 85.51, 63.1, 56.49, 50.6, 89.4, 89.49, 89.35, 88.81, 87.29, 89.35, 89.35, 89.7, 89.11, 87.44, 89.4, 89.55, 89.67, 89.32, 88.39, 89.29, 89.46, 89.7, 89.38, 88.54, 89.38, 89.2, 89.49, 89.49, 88.99, 89.29, 89.4, 89.52, 89.29, 88.9, 89.2, 88.48, 89.05, 88.36, 88.69, 89.17, 89.14, 88.36, 87.8, 88.66, 89.26, 89.02, 88.24, 87.77, 85.65, 89.29, 89.32, 88.54, 88.18, 84.52, 89.32, 89.73, 86.76, 80.6, 61.88, 89.29, 89.08, 81.37, 74.49, 77.71, 71.28, 61.13, 65.6, 83.3, 85.62, 54.52, 76.04, 68.07, 59.4, 57.92, 58.12, 87.38, 85.83, 83.6, 81.76, 79.88, 89.61, 89.67, 89.61, 89.61, 89.52, 89.49, 89.52, 89.64, 89.23, 89.38, 86.9, 56.16, 66.88, 75.8, 74.64, 84.91, 81.46, 79.17, 79.26, 77.59, 77.62, 73.15, 67.17, 63.42, 57.86]
		# MA_list = np.array(MA_list)

		MA_list = MA_list / 100.

		MA_list_nvidia_Hc_augmix = [75.72, 75.67, 75.79, 74.89, 73.26, 68.15, 75.61, 75.64, 75.62, 75.49, 75.06, 75.37, 72.84, 69.41, 66.22, 67.25, 75.67, 75.64, 75.72, 75.66, 75.81, 75.57, 75.34, 75.67, 75.51, 75.54, 75.71, 75.59, 75.42, 75.49, 75.46, 75.71, 75.46, 75.71, 75.67, 75.67, 75.66, 75.69, 75.34, 75.36, 75.37, 75.69, 75.62, 75.67, 75.41, 74.99, 75.74, 75.54, 75.69, 75.51, 75.67, 75.57, 75.57, 75.47, 75.54, 75.59, 75.56, 75.41, 75.46, 75.37, 75.27, 75.71, 75.81, 75.79, 75.51, 75.24, 75.71, 75.61, 75.46, 75.51, 74.53, 75.66, 75.51, 74.66, 73.36, 65.47, 73.44, 70.05, 69.20, 74.56, 75.26, 70.88, 75.21, 74.76, 73.86, 72.93, 71.55, 74.96, 74.64, 74.43, 74.06, 73.33, 75.87, 75.57, 75.59, 75.62, 75.81, 75.62, 75.87, 75.79, 75.72, 75.32, 74.99, 74.09, 74.11, 72.69, 73.94, 74.74, 74.29, 73.89, 74.24, 73.66, 75.64, 75.37, 74.43, 73.48, 71.38]
		MA_list_nvidia_Hc_ours = [78.32, 78.24, 78.32, 78.35, 76.96, 75.02, 78.29, 78.50, 78.39, 77.82, 76.44, 77.77, 74.44, 75.46, 76.04, 70.01, 78.40, 78.45, 78.64, 78.45, 78.77, 78.47, 78.60, 78.55, 78.24, 78.26, 78.40, 78.55, 78.65, 78.65, 78.45, 78.44, 78.62, 78.55, 78.47, 78.54, 78.37, 78.37, 78.40, 78.32, 78.04, 78.40, 78.60, 78.29, 78.27, 77.89, 78.44, 78.39, 78.44, 78.49, 78.29, 78.44, 78.26, 78.39, 78.35, 78.31, 78.45, 78.54, 78.21, 78.24, 77.71, 78.47, 78.49, 78.55, 78.22, 77.54, 78.42, 78.74, 78.04, 76.91, 58.99, 78.45, 77.99, 76.64, 75.19, 69.78, 72.41, 71.76, 75.02, 76.91, 78.21, 66.20, 78.27, 77.41, 76.64, 74.23, 72.86, 77.29, 76.76, 76.74, 76.27, 75.67, 78.31, 78.24, 78.44, 78.39, 78.31, 78.26, 78.47, 78.26, 77.87, 78.27, 76.89, 75.77, 74.91, 71.51, 73.94, 76.61, 75.52, 74.59, 74.51, 73.91, 75.46, 72.78, 67.37, 64.20, 59.27]
		
		MA_list_commaai_Hc_augmix = [68.98, 69.50, 68.93, 69.28, 66.18, 63.20, 69.65, 69.21, 67.83, 67.13, 66.22, 68.41, 59.27, 67.23, 66.97, 67.30, 69.63, 69.55, 68.85, 69.36, 68.68, 69.95, 68.73, 68.55, 68.75, 68.48, 69.60, 68.35, 68.40, 68.80, 67.75, 69.48, 69.71, 68.53, 68.53, 67.55, 68.50, 68.51, 68.53, 68.81, 68.08, 69.11, 68.51, 69.00, 67.88, 65.90, 69.41, 68.13, 69.25, 68.80, 69.06, 69.38, 69.03, 68.41, 69.11, 68.40, 68.05, 68.71, 69.46, 68.78, 68.81, 69.51, 69.11, 68.17, 68.17, 65.63, 69.40, 68.38, 67.52, 66.18, 58.08, 68.61, 68.15, 67.50, 65.85, 55.89, 53.10, 67.00, 67.88, 63.59, 68.02, 61.31, 69.28, 68.15, 68.46, 67.12, 65.77, 68.43, 67.17, 68.50, 67.72, 67.33, 68.96, 69.25, 69.00, 68.91, 68.12, 68.98, 68.58, 70.26, 69.38, 69.30, 68.78, 67.60, 65.58, 63.14, 65.88, 68.41, 67.38, 65.70, 65.30, 64.87, 65.92, 66.17, 63.25, 62.22, 57.84]
		MA_list_commaai_Hc_ours = []

		MA_list_nvidia_Ads_augmix = [82.84, 82.66, 81.89, 79.86, 71.15, 59.03, 82.66, 82.05, 81.96, 80.18, 77.85, 82.26, 70.73, 54.37, 17.32, 11.76, 82.96, 83.22, 82.00, 81.16, 78.83, 83.08, 83.10, 82.24, 81.49, 79.46, 83.03, 82.61, 82.35, 82.12, 79.55, 83.03, 83.19, 82.33, 80.91, 76.98, 82.98, 82.82, 82.59, 81.68, 78.76, 83.03, 83.38, 82.73, 82.21, 80.72, 83.19, 82.70, 82.52, 83.03, 81.84, 83.05, 82.33, 82.84, 83.01, 81.84, 83.12, 83.47, 82.31, 81.47, 79.34, 83.15, 83.31, 79.90, 77.31, 66.39, 83.01, 83.10, 82.26, 81.09, 14.01, 82.89, 82.14, 79.06, 75.58, 53.80, 70.68, 52.31, 53.62, 77.61, 77.71, 54.86, 81.56, 79.62, 75.70, 71.52, 68.02, 79.65, 77.38, 75.56, 73.95, 71.36, 82.61, 82.66, 82.28, 82.49, 82.40, 82.75, 82.47, 82.70, 81.98, 81.98, 81.07, 79.34, 77.22, 73.97, 75.63, 80.67, 78.45, 75.98, 76.17, 74.63, 80.23, 78.66, 73.20, 66.62, 54.67]
		MA_list_nvidia_Ads_ours = [95.21, 95.38, 95.10, 93.98, 91.29, 94.07, 95.42, 95.00, 94.51, 92.27, 90.71, 93.07, 73.51, 89.03, 51.49, 46.43, 95.99, 95.89, 95.45, 95.45, 95.12, 95.99, 95.80, 95.38, 95.10, 94.07, 96.03, 96.13, 96.06, 96.20, 95.28, 95.94, 96.06, 95.96, 96.15, 95.68, 95.99, 96.22, 96.17, 95.92, 94.14, 96.03, 96.13, 96.24, 96.17, 95.35, 95.77, 95.77, 95.70, 95.26, 95.31, 95.87, 95.63, 95.70, 94.65, 95.31, 96.01, 95.54, 94.07, 93.37, 91.18, 96.01, 95.68, 93.88, 93.28, 91.18, 96.15, 96.10, 92.02, 71.45, 53.06, 96.03, 95.35, 88.91, 81.58, 50.37, 64.12, 65.27, 84.97, 82.54, 94.37, 49.98, 92.27, 89.08, 83.85, 76.47, 71.15, 87.77, 84.36, 81.37, 77.94, 73.88, 95.31, 95.28, 94.98, 95.10, 95.07, 95.05, 94.93, 94.93, 94.56, 93.51, 91.57, 86.97, 83.50, 78.62, 76.05, 88.40, 85.81, 82.31, 81.96, 79.41, 72.67, 63.00, 49.84, 43.00, 33.36]

		MA_list_commaai_Ads_augmix = [60.64, 60.13, 60.18, 59.57, 53.78, 46.85, 59.71, 59.17, 57.98, 58.17, 56.47, 57.49, 37.63, 37.84, 34.06, 8.73, 60.29, 59.76, 60.04, 58.96, 52.08, 59.38, 58.68, 54.23, 51.54, 43.79, 59.13, 59.06, 55.49, 50.09, 37.30, 59.85, 59.69, 56.47, 52.08, 42.18, 59.20, 58.68, 58.10, 56.19, 52.64, 60.13, 58.43, 58.43, 56.82, 49.25, 60.67, 60.18, 58.68, 58.22, 59.15, 60.13, 58.52, 59.43, 58.87, 60.29, 59.76, 59.52, 58.66, 58.92, 57.31, 60.01, 59.92, 58.96, 57.59, 45.92, 60.57, 61.20, 58.96, 54.65, 34.10, 59.99, 57.87, 52.24, 45.82, 32.19, 38.28, 35.22, 37.61, 38.49, 55.67, 20.05, 57.66, 59.36, 56.16, 55.23, 53.76, 58.12, 56.40, 54.32, 53.66, 51.94, 59.34, 59.24, 59.24, 59.36, 60.39, 59.83, 59.31, 59.78, 58.29, 58.45, 54.30, 49.11, 46.83, 43.88, 44.07, 51.73, 49.63, 47.29, 47.53, 46.78, 48.39, 43.07, 38.68, 37.58, 32.40]
		MA_list_commaai_Ads_ours = []

		base_MA_nvidia_B = [88.36, 88.21, 88.06, 86.07, 81.16, 73.33, 88.33, 86.01, 81.42, 76.39, 73.15, 88.63, 74.97, 57.67, 48.83, 49.16, 87.81, 65.06, 57.91, 55.44, 53.24, 87.67, 61.36, 52.14, 47.44, 45.14, 88.54, 73, 53.54, 48.21, 44.16, 87.91, 69.55, 51.16, 43.66, 39.97, 88.18, 69.73, 54.25, 46.84, 43.03, 87.7, 66.22, 52.47, 47.14, 42.58, 88.09, 82.76, 63.06, 52.14, 51.28, 88.15, 69.25, 51.51, 51.33, 51.22, 88.06, 83.89, 72.56, 63.83, 58.36, 88.33, 74.46, 61.57, 56.48, 53.15, 88.51, 69.43, 54.61, 53.21, 52.63, 88.36, 70.35, 49.1, 43.21, 39.43, 59.73, 54.02, 40.89, 50.06, 54.02, 56.31, 76.4, 69.7, 62.62, 61.1, 60.33, 85.57, 83.66, 81.79, 79.97, 78.15, 88.15, 88.21, 88.04, 88.27, 88.1, 88.42, 88.01, 87.41, 85.39, 82.17, 62.77, 50.68, 54.94, 55.45, 55.33, 55.8, 52.14, 51.67, 51.67, 51.22, 58.72, 55, 52.44, 50.8, 48.12]
		base_MA_nvidia_Hc = [75.84, 75.69, 75.44, 74.51, 70.05, 61.99, 75.87, 75.36, 74.83, 72.26, 69.81, 74.91, 69.06, 66.80, 71.50, 62.79, 75.77, 69.85, 58.47, 53.33, 46.47, 75.79, 66.73, 54.43, 51.25, 45.75, 75.77, 74.76, 65.85, 60.26, 46.72, 75.71, 70.38, 55.68, 47.57, 35.03, 75.86, 72.93, 52.61, 45.07, 42.66, 75.87, 67.35, 49.95, 44.17, 36.13, 75.86, 74.23, 73.14, 68.70, 65.53, 75.86, 73.71, 66.85, 65.88, 65.50, 75.77, 75.71, 73.76, 72.56, 70.81, 75.72, 71.60, 60.99, 56.74, 50.02, 75.77, 71.53, 57.84, 51.33, 60.51, 75.72, 70.38, 54.25, 51.12, 47.15, 47.47, 48.42, 66.22, 62.85, 50.08, 57.59, 74.86, 74.11, 72.91, 70.63, 68.95, 75.07, 74.78, 74.13, 73.69, 72.93, 75.61, 75.76, 75.61, 75.64, 75.74, 75.79, 75.42, 75.41, 75.44, 74.41, 71.74, 67.85, 67.28, 66.05, 64.80, 64.67, 60.04, 58.04, 58.36, 56.98, 61.85, 58.21, 53.90, 51.95, 49.15]
		base_MA_nvidia_Ads = [91.08, 89.82, 88.59, 82.77, 68.88, 53.59, 89.50, 85.97, 69.65, 56.93, 47.25, 88.77, 49.30, 38.89, 4.93, 0.00, 91.53, 86.72, 76.05, 71.64, 63.54, 91.53, 85.27, 74.16, 69.70, 60.50, 79.51, 24.07, 6.00, 2.01, 0.33, 88.28, 39.15, 24.67, 20.61, 4.55, 81.14, 26.26, 6.26, 2.10, 0.23, 88.54, 39.43, 25.26, 21.15, 13.87, 87.98, 83.38, 64.17, 59.55, 33.71, 89.36, 52.85, 31.35, 49.30, 33.71, 91.81, 72.81, 49.14, 43.25, 30.95, 77.43, 23.76, 6.89, 6.44, 2.68, 67.83, 8.50, 0.37, 0.14, 0.00, 88.38, 34.34, 20.61, 11.18, 2.15, 15.92, 0.00, 28.99, 2.12, 37.07, 1.26, 86.04, 80.37, 71.08, 63.52, 57.77, 78.52, 74.39, 70.28, 67.81, 63.82, 89.73, 89.50, 90.15, 90.10, 89.96, 89.94, 89.26, 87.32, 81.61, 74.67, 18.53, 11.81, 11.83, 5.77, 2.57, 14.43, 10.85, 10.22, 11.06, 10.57, 12.89, 11.93, 9.85, 9.69, 8.43]
		base_MA_commaai_B = [81.96, 82.26, 82.50, 80.86, 77.56, 70.89, 82.71, 80.54, 77.74, 72.71, 67.68, 73.54, 49.49, 33.33, 54.32, 52.77, 81.76, 64.61, 51.73, 48.51, 46.67, 82.11, 57.95, 46.76, 45.57, 49.43, 82.41, 65.48, 51.93, 50.00, 47.38, 81.85, 66.04, 58.04, 57.41, 56.25, 82.65, 71.31, 60.51, 57.32, 50.95, 82.62, 70.89, 59.55, 58.21, 53.18, 81.43, 74.46, 66.19, 61.52, 53.84, 82.32, 67.80, 50.74, 53.04, 53.24, 83.15, 81.55, 73.15, 68.93, 57.14, 81.70, 74.11, 62.14, 58.27, 52.41, 82.86, 72.71, 47.02, 39.94, 37.68, 82.29, 76.25, 63.90, 60.45, 54.85, 54.64, 32.50, 36.88, 48.99, 60.65, 57.23, 82.23, 79.52, 78.07, 75.80, 73.45, 81.52, 80.18, 79.37, 78.01, 76.85, 82.35, 83.04, 82.62, 82.29, 83.01, 82.50, 82.11, 81.40, 80.15, 77.92, 66.99, 59.20, 57.92, 57.68, 57.23, 61.46, 60.51, 59.46, 60.18, 59.52, 62.02, 58.33, 55.74, 54.79, 52.38]
		base_MA_commaai_Hc = [74.23, 72.58, 73.46, 71.61, 67.57, 61.85, 73.53, 71.94, 70.63, 67.87, 63.47, 71.71, 69.08, 70.35, 67.50, 71.11, 74.08, 68.66, 57.09, 52.91, 47.09, 72.66, 51.53, 33.60, 31.92, 31.87, 73.76, 65.32, 46.35, 42.22, 39.09, 73.14, 66.23, 47.30, 42.39, 36.36, 73.41, 69.43, 58.54, 53.33, 45.09, 73.71, 56.33, 29.62, 20.13, 10.31, 73.16, 68.32, 65.73, 61.29, 50.52, 73.96, 70.28, 60.31, 58.26, 50.35, 72.66, 73.38, 71.96, 71.35, 67.02, 72.73, 71.06, 62.02, 58.28, 51.32, 72.86, 72.61, 63.44, 60.89, 65.07, 72.44, 66.53, 38.41, 26.26, 15.85, 69.68, 68.80, 69.91, 63.80, 36.70, 67.35, 72.96, 72.49, 70.90, 69.66, 68.61, 71.74, 70.00, 70.10, 69.65, 69.68, 72.39, 73.13, 73.31, 73.01, 72.93, 73.14, 71.99, 72.98, 72.64, 72.08, 64.44, 47.40, 47.97, 41.03, 30.69, 45.75, 32.88, 28.01, 28.84, 26.96, 53.53, 50.27, 46.15, 46.30, 43.89]
		base_MA_commaai_Ads = [79.34, 78.34, 78.08, 76.14, 67.81, 54.53, 78.76, 77.24, 72.99, 64.31, 54.95, 39.54, 5.72, 8.96, 3.06, 2.33, 77.33, 38.47, 4.20, 2.01, 1.31, 77.80, 49.56, 30.56, 29.74, 30.42, 78.27, 41.99, 4.55, 1.42, 0.63, 78.78, 62.98, 45.19, 40.64, 31.72, 75.00, 35.41, 22.53, 21.03, 17.79, 78.97, 32.66, 14.38, 9.92, 5.35, 78.66, 75.68, 59.59, 48.04, 44.89, 77.71, 70.17, 43.58, 32.96, 45.80, 78.06, 74.53, 63.94, 58.01, 46.17, 76.75, 40.45, 12.09, 9.64, 10.41, 73.83, 18.79, 1.73, 1.70, 1.82, 78.01, 43.77, 12.32, 4.53, 0.96, 11.41, 1.52, 6.82, 1.61, 33.59, 1.33, 77.66, 75.19, 71.36, 67.62, 61.74, 72.90, 70.28, 66.53, 64.17, 59.45, 78.22, 79.32, 78.36, 77.59, 78.38, 78.71, 77.92, 77.17, 76.28, 71.71, 25.75, 14.40, 14.85, 10.60, 6.68, 15.92, 11.95, 9.76, 10.74, 9.41, 21.66, 18.51, 18.09, 18.02, 18.65]
		if "commaai" in train_folder:
			if "trainB" in train_folder:
				base_MA = np.array(base_MA_commaai_B) / 100.
			elif "trainHc" in train_folder:
				base_MA = np.array(base_MA_commaai_Hc) / 100.
			elif "trainAds" in train_folder:
				base_MA = np.array(base_MA_commaai_Ads) / 100.
		else:
			if "trainB" in train_folder:
				base_MA = np.array(base_MA_nvidia_B) / 100.
			elif "trainHc" in train_folder:
				base_MA = np.array(base_MA_nvidia_Hc) / 100.
			elif "trainAds" in train_folder:
				base_MA = np.array(base_MA_nvidia_Ads) / 100.

		corruption_accs_1, corruption_accs_2, corruption_accs_3, corruption_accs_4 = reformal_corruption_accs(MA_list)
		base_accs_1, base_accs_2, base_accs_3, base_accs_4 = reformal_corruption_accs(base_MA)

		# corruption_accs_1, corruption_accs_2, corruption_accs_3, corruption_accs_4 = reformal_corruption_accs(np.array(MA_list_commaai_Ads_augmix) / 100.)
		# base_accs_1, base_accs_2, base_accs_3, base_accs_4 = reformal_corruption_accs(np.array(base_MA_commaai_Ads) / 100.)

		mce1 = compute_mce(corruption_accs_1, base_accs_1)
		mce2 = compute_mce(corruption_accs_2, base_accs_2)
		mce3 = compute_mce(corruption_accs_3, base_accs_3)
		mce4 = compute_mce(corruption_accs_4, base_accs_4)

		print("mCE1", mce1)
		print("mCE2", mce2)
		print("mCE3", mce3)
		print("mCE4", mce4)



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
	pytorch_flag = True
	suffix = ""
	#pretrain_model = "tmp"
	#pretrain_model_path = TRAIN_OUTPUT_ROOT + pretrain_model + "/model-final.h5"
	pretrain_model_path = ""

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
	elif subtask_id == '38':
		TRAIN_FOLDER_LIST = [["trainAudi1", "trainAudi1segall"]]
		VAL_LIST = [["valAudi1", "valAudi1segall"]]
		TRAIN_RATIO_LIST = [[1,1]]
		pack_in_channel = True
	elif subtask_id == '39':
		TRAIN_FOLDER_LIST = [["trainAudi2", "trainAudi2segall"]]
		VAL_LIST = [["valAudi2", "valAudi2segall"]]
		TRAIN_RATIO_LIST = [[1,1]]
		pack_in_channel = True
	elif subtask_id == '40':
		TRAIN_FOLDER_LIST = [["trainAudi1", "trainAudi1segall"]]
		VAL_LIST = [["valAudi1", "valAudi1segall"]]
		TRAIN_RATIO_LIST = [[1,1]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_39_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '41':
		TRAIN_FOLDER_LIST = [["trainAudi2", "trainAudi2segall"]]
		VAL_LIST = [["valAudi2", "valAudi2segall"]]
		TRAIN_RATIO_LIST = [[1,1]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_38_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '42':
		TRAIN_FOLDER_LIST = [["trainAudi1"]]
		VAL_LIST = [["valAudi1"]]
		TRAIN_RATIO_LIST = [[0.2]]
		pack_in_channel = True
	elif subtask_id == '43':
		TRAIN_FOLDER_LIST = [["trainAudi2"]]
		VAL_LIST = [["valAudi2"]]
		TRAIN_RATIO_LIST = [[0.2]]
		pack_in_channel = True
	elif subtask_id == '44':
		TRAIN_FOLDER_LIST = [["trainAudi1"]]
		VAL_LIST = [["valAudi1"]]
		TRAIN_RATIO_LIST = [[0.2]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "trainAudi2_pytorch_slabel/model-final.pth"
	elif subtask_id == '45':
		TRAIN_FOLDER_LIST = [["trainAudi2"]]
		VAL_LIST = [["valAudi2"]]
		TRAIN_RATIO_LIST = [[0.2]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "trainAudi1_pytorch_slabel/model-final.pth"
	elif subtask_id == '46':
		TRAIN_FOLDER_LIST = [["trainAudi3"]]
		VAL_LIST = [["valAudi3"]]
		TRAIN_RATIO_LIST = [[0.2]]
		# pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "trainAudi4_pytorch/model-final.pth"
	elif subtask_id == '47':
		TRAIN_FOLDER_LIST = [["trainAudi4"]]
		VAL_LIST = [["valAudi4"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		# pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "trainAudi3_pytorch/model-final.pth"
	elif subtask_id == '48':
		TRAIN_FOLDER_LIST = [["trainAudi5"]]
		VAL_LIST = [["valAudi5"]]
		VAL_LIST = [["trainAudi5"],["trainAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
	elif subtask_id == '49':
		TRAIN_FOLDER_LIST = [["trainAudi6"]]
		VAL_LIST = [["valAudi6"]]
		VAL_LIST = [["trainAudi5"],["trainAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
	elif subtask_id == '50':
		TRAIN_FOLDER_LIST = [["trainAudi5"]]
		VAL_LIST = [["valAudi5"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		VAL_LIST = [["trainAudi5"],["trainAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 7
	elif subtask_id == '51':
		TRAIN_FOLDER_LIST = [["trainAudi6"]]
		VAL_LIST = [["valAudi6"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		VAL_LIST = [["trainAudi5"],["trainAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 7
	elif subtask_id == '52':
		TRAIN_FOLDER_LIST = [["trainAudi5"]]
		VAL_LIST = [["valAudi5"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		VAL_LIST = [["trainAudi5"],["trainAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 8
	elif subtask_id == '53':
		TRAIN_FOLDER_LIST = [["trainAudi6"]]
		VAL_LIST = [["valAudi6"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		VAL_LIST = [["trainAudi5"],["trainAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 8
	elif subtask_id == '54':
		TRAIN_FOLDER_LIST = [["trainAudi5"]]
		VAL_LIST = [["valAudi5"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 7
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_51_1_BN7_pytorch/model-final.pth"
	elif subtask_id == '55':
		TRAIN_FOLDER_LIST = [["trainAudi6"]]
		VAL_LIST = [["valAudi6"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 7
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_50_1_BN7_pytorch/model-final.pth"
	elif subtask_id == '56':
		TRAIN_FOLDER_LIST = [["trainAudi5"]]
		VAL_LIST = [["valAudi5"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 8
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_53_1_BN8_pytorch/model-final.pth"
	elif subtask_id == '57':
		TRAIN_FOLDER_LIST = [["trainAudi6"]]
		VAL_LIST = [["valAudi6"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		BN_flag = 8
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_52_1_BN8_pytorch/model-final.pth"
	elif subtask_id == '58':
		TRAIN_FOLDER_LIST = [["trainAudi5"]]
		VAL_LIST = [["valAudi5"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_49_1_pytorch/model-final.pth"
	elif subtask_id == '59':
		TRAIN_FOLDER_LIST = [["trainAudi6"]]
		VAL_LIST = [["valAudi6"]]
		VAL_LIST = [["valAudi5"],["valAudi6"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_48_1_pytorch/model-final.pth"
	elif subtask_id == '60':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5segall"]]
		VAL_LIST = [["valAudi5", "valAudi5segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
	elif subtask_id == '61':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6segall"]]
		VAL_LIST = [["valAudi6", "valAudi6segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
	elif subtask_id == '62':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5segall"]]
		VAL_LIST = [["valAudi5", "valAudi5segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
	elif subtask_id == '63':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6segall"]]
		VAL_LIST = [["valAudi6", "valAudi6segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
	elif subtask_id == '64':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5segall"]]
		VAL_LIST = [["valAudi5", "valAudi5segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
	elif subtask_id == '65':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6segall"]]
		VAL_LIST = [["valAudi6", "valAudi6segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
	elif subtask_id == '66':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5segall"]]
		VAL_LIST = [["valAudi5", "valAudi5segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_61_1_pack_channel_pytorch/model-final.pth"
		# BN_flag = 8
	elif subtask_id == '67':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6segall"]]
		VAL_LIST = [["valAudi6", "valAudi6segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_60_1_pack_channel_pytorch/model-final.pth"
		# BN_flag = 8
	elif subtask_id == '68':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5segall"]]
		VAL_LIST = [["valAudi5", "valAudi5segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_63_1_BN7_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '69':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6segall"]]
		VAL_LIST = [["valAudi6", "valAudi6segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_62_1_BN7_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '70':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5segall"]]
		VAL_LIST = [["valAudi5", "valAudi5segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_65_1_BN8_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '71':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6segall"]]
		VAL_LIST = [["valAudi6", "valAudi6segall"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_64_1_BN8_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '72':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
	elif subtask_id == '73':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6seginfer"]]
		VAL_LIST = [["valAudi6", "valAudi6seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
	elif subtask_id == '74':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
	elif subtask_id == '75':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6seginfer"]]
		VAL_LIST = [["valAudi6", "valAudi6seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
	elif subtask_id == '76':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
	elif subtask_id == '77':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6seginfer"]]
		VAL_LIST = [["valAudi6", "valAudi6seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
	elif subtask_id == '78':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_73_1_pack_channel_pytorch/model-final.pth"
		# BN_flag = 8
	elif subtask_id == '79':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6seginfer"]]
		VAL_LIST = [["valAudi6", "valAudi6seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_72_1_pack_channel_pytorch/model-final.pth"
		# BN_flag = 8
	elif subtask_id == '80':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_75_1_BN7_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '81':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6seginfer"]]
		VAL_LIST = [["valAudi6", "valAudi6seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 7
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_74_1_BN7_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '82':
		TRAIN_FOLDER_LIST = [["trainAudi5", "trainAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_77_1_BN8_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '83':
		TRAIN_FOLDER_LIST = [["trainAudi6", "trainAudi6seginfer"]]
		VAL_LIST = [["valAudi6", "valAudi6seginfer"]]
		VAL_LIST = [["valAudi5", "valAudi5seginfer"], ["valAudi6", "valAudi6seginfer"]]
		TRAIN_RATIO_LIST = [[0.2],[1]]
		pack_in_channel = True
		BN_flag = 8
		pretrain_model_path = TRAIN_OUTPUT_ROOT + "combine_76_1_BN8_pack_channel_pytorch/model-final.pth"
	elif subtask_id == '100':
		#For test
		TRAIN_FOLDER_LIST = [["trainB_blur"]]
		TRAIN_RATIO_LIST = [[1]]
		VAL_LIST = [["trainAudi1", "trainAudi1segall"]]
	else:
		print('invalid subtask_id!!!')
		return

	if BN_flag > 0:
		suffix = suffix + "_BN" + str(BN_flag)
	if pack_in_channel:
		suffix = suffix + "_pack_channel"
	if pytorch_flag:
		suffix = suffix + "_pytorch"

	i = 0
	res_array = []
	for train_folder_list in TRAIN_FOLDER_LIST:
		for train_ratio in TRAIN_RATIO_LIST:

			trainOurputFolder = "combine_" + str(subtask_id) + "_" + str(i) + suffix
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
			if pytorch_flag:
				modelPath = trainOutputPath + "model-final.pth"

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
				res = test_network_multi(modelPath, imagePath_list, labelPath_list, valOutputPath, BN_flag=BN_flag, pack_flag=pack_in_channel, pytorch_flag=pytorch_flag)
				res_array.append(res)
			i += 1

	for res in res_array:
		print(res*100)

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
			single_test_2_streams(args.subtask_id)
		else:
			print("Unknown task: " + args.task_id)
	else:
		# single_test()
		single_test_with_config('100')
		#single_test_AdvProp()
		#single_test_ImgAndFeature()
		#single_test_2_streams()
		#multi_factor_search_test()
		#unit_test_for_style()
		#unit_test_for_quality()
		#combination_test_for_style('36')
		#combination_test_for_style_pretrain()
		#test_AdvProp()
		#calculate_FID_folder('11')
		#calculate_FID_list_files_succ_fail('1')
		#calculate_FID_list_files()
		#visualize_model_on_image()
		#calculate_L2D()