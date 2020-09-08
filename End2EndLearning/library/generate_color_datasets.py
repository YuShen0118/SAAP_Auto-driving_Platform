#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:51:06 2020

@author: laurazheng
"""

import cv2, os, glob
from pathlib import Path
import matplotlib.pyplot as plt

dataFolder = "/media/yushen/workspace2/projects/SAAP_Auto-driving_Platform/Data/udacityA_nvidiaB/valB"
# outputPath = "/Users/laurazheng/Desktop/udacityA_nvidiaB"

def generate_RGB_dataset(originalDataset, channel, direction):

    color_str_dic = {
        0: "B",
        1: "G", 
        2: "R"
    }
    
    color_str = color_str_dic.get(channel)
           
    direction_str = "darker" if direction == 0 else "lighter"
    
    saveDir = "_".join([dataFolder, color_str, direction_str])
    
    for i in glob.glob(os.path.join(originalDataset, "*.jpg")):
        temp = cv2.imread(str(i))
        image = temp.copy()
        
        if direction == 0:
            image[:, :, channel] = image[:, :, channel] * 0.5
        elif direction == 1:
            image[:, :, channel] = (image[:, :, channel] * 0.5) + (255 * 0.5)

        
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            
        saveAsName = os.path.join(saveDir, os.path.basename(i))
        
        cv2.imwrite(saveAsName, image)
        
def generate_HSV_datasets(originalDataset, channel, direction):
    
    color_str_dic = {
        0: "H",
        1: "S", 
        2: "V"
    }
    
    color_str = color_str_dic.get(channel)
           
    direction_str = "lighter" if direction == 0 else "darker"
    
    saveDir = "_".join([dataFolder, color_str, direction_str])
    
    for i in glob.glob(os.path.join(originalDataset, "*.jpg")):
        temp = cv2.imread(i)
        image = temp.copy()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if direction == 0:
            image[:, :, channel] = image[:, :, channel] * 0.5
        elif direction == 1:
            if channel == 0:
                image[:, :, channel] = (image[:, :, channel] * 0.5) + (180 * 0.5)
            else:
                image[:, :, channel] = (image[:, :, channel] * 0.5) + (255 * 0.5)

        
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            
        saveAsName = os.path.join(saveDir, os.path.basename(i))
        
        cv2.imwrite(saveAsName, image)
        
        
# Generating dataset modifying blue channel
generate_RGB_dataset(dataFolder, 0, 0)
generate_RGB_dataset(dataFolder, 0, 1)

# Generating dataset modifying green channel
generate_RGB_dataset(dataFolder, 1, 0)
generate_RGB_dataset(dataFolder, 1, 1)

# Generating dataset modifying red channel
generate_RGB_dataset(dataFolder, 2, 0)
generate_RGB_dataset(dataFolder, 2, 1)
        
# Generating dataset modifying hue
generate_HSV_datasets(dataFolder, 0, 0)
generate_HSV_datasets(dataFolder, 0, 1)

# Generating dataset modifying saturation
generate_HSV_datasets(dataFolder, 1, 0)
generate_HSV_datasets(dataFolder, 1, 1)

# Generating dataset modifying value
generate_HSV_datasets(dataFolder, 2, 0)
generate_HSV_datasets(dataFolder, 2, 1)
