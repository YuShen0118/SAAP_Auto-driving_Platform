from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
from learning import filter_dataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print('Platform root: ', ROOT_DIR)
#root = ROOT_DIR + '/Data/udacityA_nvidiaB/'
#print('Dataset root: ', root)

DATASET_ROOT = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB/")
OUTPUT_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB_results/"
TRAIN_OUTPUT_ROOT = OUTPUT_ROOT + "train_results/"
#dataset_path = os.path.join("C:/projects/SAAP_Auto-driving_Platform/Data/nvidia/")
#dataset_path = os.path.join("/media/yushen/workspace/projects/SAAP_Auto-driving_Platform/Data/nvidia/")

def get_label_file_name(folder_name):
    if "valA" in folder_name:
        labelName = "labelsA_val.csv"
    elif "valB" in folder_name:
        labelName = "labelsB_val.csv"
    elif "valC1" in folder_name:
        labelName = "labelsC1_val.csv"
    elif "valC2" in folder_name:
        labelName = "labelsC2_val.csv"
    elif "trainA" in folder_name:
        labelName = "labelsA_train.csv"
    elif "trainB" in folder_name:
        labelName = "labelsB_train.csv"
    elif "trainC1" in folder_name:
        labelName = "labelsC1_train.csv"
    elif "trainC2" in folder_name:
        labelName = "labelsC2_train.csv"
    return labelName

def filter_dataset_outer(modelPath, imagePath, labelPath, outputPath, filter_percent=0.1):
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
    BN_flag = 0
    target_BN_folder = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB_results/test_results/(trainB)_(valB)/")
    filter_dataset(modelPath, imagePath, labelPath, outputPath, netType, flags, specs, BN_flag, target_BN_folder, filter_percent)


if __name__ == '__main__':
    train_folder = "trainB"
    #val_folder = "valB"
    val_folder = "trainC1"

    modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"

    imagePath = DATASET_ROOT + val_folder + "/"
    labelName = get_label_file_name(val_folder)
    labelPath = DATASET_ROOT + labelName
    #labelPath = DATASET_ROOT + "labelsB_train.csv"

    outputPath = DATASET_ROOT + "labelsC1_train_similarBN10.csv"
    #outputPath = DATASET_ROOT + "test.csv"
    #modelPath = ""
    filter_dataset_outer(modelPath, imagePath, labelPath, outputPath, filter_percent=0.1)