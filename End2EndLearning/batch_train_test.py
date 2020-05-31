### This script is the main training file.

import sys
import os

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from train import train_network, train_network_multi
from test import test_network

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

def single_test():
	train_folder = "trainA_MUNIT_GAN"
	val_folder = "valB"

	imagePath = DATASET_ROOT + train_folder + "/"
	if "trainA" in train_folder:
		labelPath = DATASET_ROOT + "labelsA_train.csv"
	else:
		labelPath = DATASET_ROOT + "labelsB_train.csv"
	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
	train_network(imagePath, labelPath, outputPath)

	modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	if "valA" in val_folder:
		labelPath = DATASET_ROOT + "labelsA_val.csv"
	else:
		labelPath = DATASET_ROOT + "labelsB_val.csv"
	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
	test_network(modelPath, imagePath, labelPath, outputPath)


def unit_test_for_style():
	TRAIN_LIST = ["trainA", "trainA_fake_GAN", "trainA_fake_color", "trainB", "trainB_fake_GAN", "trainB_fake_color"]
	VAL_LIST = ["valA", "valA_fake_GAN", "valA_fake_color", "valB", "valB_fake_GAN", "valB_fake_color"]

	for train_folder in TRAIN_LIST:
		imagePath = DATASET_ROOT + train_folder + "/"
		if "trainA" in train_folder:
			labelPath = DATASET_ROOT + "labelsA_train.csv"
		else:
			labelPath = DATASET_ROOT + "labelsB_train.csv"
		outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
		train_network(imagePath, labelPath, outputPath)

		for val_folder in VAL_LIST:
			modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"
			val_folder = val_folder.replace("train", "val")

			imagePath = DATASET_ROOT + val_folder + "/"
			if "valA" in val_folder:
				labelPath = DATASET_ROOT + "labelsA_val.csv"
			else:
				labelPath = DATASET_ROOT + "labelsB_val.csv"
			outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
			test_network(modelPath, imagePath, labelPath, outputPath)

def unit_test_for_quality(subtask_id):
	TRAIN_LIST_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3"],
					["trainB", "trainB_noise_1", "trainB_noise_2", "trainB_noise_3"],
					["trainB", "trainB_distort_1", "trainB_distort_2", "trainB_distort_3"]]

	if subtask_id == '0':
		TRAIN_LIST_LIST = [["trainB_blur_4", "trainB_blur_5"]]
	elif subtask_id == '1':
		TRAIN_LIST_LIST = [["trainB_noise_4", "trainB_noise_5"]]
	elif subtask_id == '2':
		TRAIN_LIST_LIST = [["trainB_distort_4", "trainB_distort_5", "trainB_distort_6"]]
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
				val_folder = val_folder.replace("train", "val")

				imagePath = DATASET_ROOT + val_folder + "/"
				labelPath = DATASET_ROOT + "labelsB_val.csv"
				outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
				test_network(modelPath, imagePath, labelPath, outputPath)

def combination_test_for_style():
	TRAIN_FOLDER_LIST = [["trainB", "trainA"],
					["trainB", "trainA_fake_GAN"],
					["trainB", "trainA_fake_color"],
					["trainB", "trainA", "trainA_fake_GAN", "trainA_fake_color"]]
	VAL_LIST = ["valB", "valA"]

	i = 0
	for train_folder_list in TRAIN_FOLDER_LIST:
		imagePath_list = []
		labelPath_list = []
		trainOutputPath = TRAIN_OUTPUT_ROOT + "combine" + str(i) + "/"

		for train_folder in train_folder_list:
			imagePath = DATASET_ROOT + train_folder + "/"
			if "trainA" in train_folder:
				labelPath = DATASET_ROOT + "labelsA_train.csv"
			else:
				labelPath = DATASET_ROOT + "labelsB_train.csv"
			imagePath_list.append(imagePath)
			labelPath_list.append(labelPath)
		train_network_multi(imagePath_list, labelPath_list, trainOutputPath)

		for val_folder in VAL_LIST:
			modelPath = trainOutputPath + "model-final.h5"
			imagePath = DATASET_ROOT + val_folder + "/"
			if "valA" in val_folder:
				labelPath = DATASET_ROOT + "labelsA_val.csv"
			else:
				labelPath = DATASET_ROOT + "labelsB_val.csv"
			valOutputPath = TEST_OUTPUT_ROOT + "(combine" + str(i) + ")_(" + val_folder + ")/test_result.txt"
			test_network(modelPath, imagePath, labelPath, valOutputPath)

		i += 1

def combination_test_for_style_pretrain():
	#TRAIN_RATIO_LIST = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
	TRAIN_RATIO_LIST = [0.25, 0.5, 0.75]
	#PRETRAIN_MODEL_LIST = ["trainA"]
	#PRETRAIN_MODEL_LIST = ["combine0"]
	PRETRAIN_MODEL_LIST = [""]
	#TRAIN_LIST = ["trainB"]
	TRAIN_LIST = ["trainA"]
	#VAL_LIST = ["valB", "valA"]
	VAL_LIST = ["valA"]
	partial_preModel = False

	id = 0
	for train_ratio in TRAIN_RATIO_LIST:
		for pretrain_model in PRETRAIN_MODEL_LIST:
			for train_folder in TRAIN_LIST:
				id += 1
				print(id)
				imagePath = DATASET_ROOT + train_folder + "/"
				if "trainA" in train_folder:
					labelPath = DATASET_ROOT + "labelsA_train.csv"
				else:
					labelPath = DATASET_ROOT + "labelsB_train.csv"

				if pretrain_model == "":
					trainOutputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + str(train_ratio) + "/"
					pretrain_model_path = ""
				else:
					if partial_preModel:
						trainOutputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + str(train_ratio) + "_(" + pretrain_model + "_partialpretrain)/"
					else:
						trainOutputPath = TRAIN_OUTPUT_ROOT + train_folder + "_" + str(train_ratio) + "_(" + pretrain_model + "_pretrain)/"
					pretrain_model_path = TRAIN_OUTPUT_ROOT + pretrain_model + "/model-final.h5"

				if (train_ratio == 1 and pretrain_model == ""):
					continue

				train_network(imagePath, labelPath, trainOutputPath, pretrain_model_path, trainRatio=train_ratio, partialPreModel=partial_preModel)

				for val_folder in VAL_LIST:
					modelPath = trainOutputPath + "/model-final.h5"
					val_folder = val_folder.replace("train", "val")

					imagePath = DATASET_ROOT + val_folder + "/"
					if "valA" in val_folder:
						labelPath = DATASET_ROOT + "labelsA_val.csv"
					else:
						labelPath = DATASET_ROOT + "labelsB_val.csv"
					outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + "_" + str(train_ratio) + "_(" + pretrain_model + "_pretrain)" + ")_(" + val_folder + ")/test_result.txt"
					test_network(modelPath, imagePath, labelPath, outputPath)

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

	if args.task_id:
		if args.task_id == '0':
			single_test()
		elif args.task_id == '1':
			unit_test_for_style()
		elif args.task_id == '2':
			unit_test_for_quality(args.subtask_id)
		elif args.task_id == '3':
			combination_test_for_style()
		elif args.task_id == '4':
			combination_test_for_style_pretrain()
		else:
			print("Unknown task: " + args.task_id)
	else:
		single_test()
		#unit_test_for_style()
		#unit_test_for_quality()
		#combination_test_for_style()
		#combination_test_for_style_pretrain()