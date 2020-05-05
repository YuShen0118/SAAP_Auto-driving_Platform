# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:58:29 2020

@author: Laura Zheng
"""
import os
import csv
import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_steering_angle_dist(dataFilePath, label):
    ''' Plots the distribution for a single steering angle label file. 
        Does not save the image to disk.
    '''
    
    steeringAngleValues = []
    title="Steering Angle Distribution Plot"
    
    with open(dataFilePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
    
        for row in csv_reader:
            steeringAngleValues.append(float(row[3]))
        
    data = np.array(steeringAngleValues)
    
    sns.set(color_codes=True)
    
    sns.distplot(data).set_title(title);

    minv = np.min(data)
    maxv = np.max(data)

    #print(minv)
    #print(maxv)

    return minv, maxv

def plot_all_datasets(NVIDIA_labels, Udacity_labels, custom_labels, outputPath):
    ''' Plots all dataset steering angle distributions and saves to the disk as PNG.
    '''
    minv_Udacity, maxv_Udacity = plot_steering_angle_dist(Udacity_labels, label="Udacity")
    minv_NVIDIA, maxv_NVIDIA = plot_steering_angle_dist(NVIDIA_labels, label="NVIDIA")
    minv_custom, maxv_custom = plot_steering_angle_dist(custom_labels, label="Custom")

    minv = min(minv_Udacity, min(minv_NVIDIA, minv_custom))
    maxv = max(maxv_Udacity, max(maxv_NVIDIA, maxv_custom))
    #print(minv)
    #print(maxv)
    #plt.xlim(-15, 15)
    plt.xlim(minv, maxv)
    plt.legend(title='Dataset', loc='upper right', labels=['Udacity', 'NVIDIA', 'Custom'])
    plt.savefig(outputPath, format='png',dpi=1200)


ROOT_DIR = os.path.abspath("../")
print('Platform root: ', ROOT_DIR)

NVIDIA_labels = ROOT_DIR + '/Data/udacityA_nvidiaB/labelsB_trainval.csv'
Udacity_labels = ROOT_DIR + '/Data/udacityA_nvidiaB/labelsA_trainval.csv'
custom_labels = ROOT_DIR + '/Data/training_simu_1/end2endLabels.csv'
output_path = ROOT_DIR + '/Data/output.png'

plot_all_datasets(NVIDIA_labels, Udacity_labels, custom_labels, output_path)
