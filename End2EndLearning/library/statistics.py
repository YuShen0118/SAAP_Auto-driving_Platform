# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:58:29 2020

@author: Laura Zheng
"""

import csv
import numpy as np
import pandas as pd
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

def plot_all_datasets(NVIDIA_labels, Udacity_labels, custom_labels, outputPath):
    ''' Plots all dataset steering angle distributions and saves to the disk as PNG.
    '''
    plt.xlim(-15, 15)
    plot_steering_angle_dist(Udacity_labels, label="Udacity")
    plot_steering_angle_dist(NVIDIA_labels, label="NVIDIA")
    plot_steering_angle_dist(custom_labels, label="Custom")
    plt.legend(title='Dataset', loc='upper right', labels=['Udacity', 'NVIDIA', 'Custom'])
    plt.savefig(outputPath, format='png',dpi=1200)


NVIDIA_labels = 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/NVIDIA/formatted_labels.csv'
Udacity_labels = 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/Udacity/track1data/formatted_labels.csv'
custom_labels = 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/training_simu_1/end2endLabels.csv'

plot_all_datasets(NVIDIA_labels, Udacity_labels, custom_labels, 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/output.png')
