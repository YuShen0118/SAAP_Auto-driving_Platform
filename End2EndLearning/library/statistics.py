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
from networks_pytorch import create_nvidia_network_pytorch
import torch


ROOT_DIR = os.path.abspath("../")
print('Platform root: ', ROOT_DIR)

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

    '''
    count=0
    for v in steeringAngleValues:
        if v < 0.5 and v > -0.5:
            count += 1

    print(str(count) + " / " + str(len(steeringAngleValues)) + " = " + str(count/len(steeringAngleValues)))
    '''
    
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
    #minv_custom, maxv_custom = plot_steering_angle_dist(custom_labels, label="Custom")

    #minv = min(minv_Udacity, min(minv_NVIDIA, minv_custom))
    #maxv = max(maxv_Udacity, max(maxv_NVIDIA, maxv_custom))

    minv = min(minv_Udacity, minv_NVIDIA)
    maxv = max(maxv_Udacity, maxv_NVIDIA)
    #print(minv)
    #print(maxv)
    #plt.xlim(-15, 15)
    plt.xlim(minv, maxv)
    plt.legend(title='Dataset', loc='upper right', labels=['Udacity', 'NVIDIA', 'Custom'])
    plt.savefig(outputPath, format='png',dpi=1200)


def draw_label_distribution():
    ROOT_DIR = os.path.abspath("../")
    print('Platform root: ', ROOT_DIR)

    NVIDIA_labels = ROOT_DIR + '/Data/udacityA_nvidiaB/labelsAudi1_train.csv'
    Udacity_labels = ROOT_DIR + '/Data/udacityA_nvidiaB/labelsAudi2_train.csv'
    custom_labels = ROOT_DIR + '/Data/training_simu_1/end2endLabels.csv'
    output_path = ROOT_DIR + '/Data/output_Audi1_2.png'

    plot_all_datasets(NVIDIA_labels, Udacity_labels, custom_labels, output_path)


def read_float_list(file_name):
    x = []
    file_in = open(file_name, 'r')
    for y in file_in.read().split('\n'):
        if len(y) > 0:
            x.append(float(y))
    return np.array(x)

def plot_distribution(data, output_folder="", title="distribution"):
    sns.set(color_codes=True)
    sns.distplot(data).set_title(title)
    minv = np.min(data)
    maxv = np.max(data)
    plt.xlim(minv, maxv)
    #plt.legend(loc='upper right', labels=BN_folders)
    plt.savefig(output_folder + title + ".png", format='png')
    plt.clf()
    #plt.show()

def show_statistics(data, base_data, output_folder="", case_title="", BN_flag=0):
    f_BN = open(output_folder + "BN_" + case_title + ".txt",'w')

    s = 0
    if BN_flag==0:
        dimension_in_each_layer = [3, 24, 24, 36, 36, 48, 48, 64, 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1]
        wanted_layer_id = [0, 1, 3, 5, 7, 9, 12, 14, 16, 18]
    elif BN_flag==1:
        dimension_in_each_layer = [3, 24, 24, 24, 36, 36, 36, 48, 48, 48, 64, 64, 64, 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        wanted_layer_id = [0, 2, 5, 8, 11, 14, 18, 21, 24, 26]
    elif BN_flag==2:
        dimension_in_each_layer = [3, 3, 24, 24, 24, 24, 24, 24, 36, 36, 36, 36, 36, 36, 48, 48, 48, 48, 48, 48, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        wanted_layer_id = [0, 4, 10, 16, 22, 28, 36, 42, 48, 52]

    data_wanted = []
    data_wanted_base = []
    value_list = []
    value_percent_list = []
    #dimension_in_each_layer = [24, 36, 48, 64, 64]
    for i in range(len(dimension_in_each_layer)):
        t = s + dimension_in_each_layer[i]
        if i in wanted_layer_id:
            print("s " + str(s) + " t " + str(t))
            l2_norm = np.linalg.norm(data[s:t])
            l2_norm_base = np.linalg.norm(base_data[s:t])
            value_list.append(l2_norm)
            value_percent_list.append(l2_norm / l2_norm_base * 100)
            print(case_title + ", layer "+str(i) + " L2 norm: " + str(l2_norm) + "  ratio: " + str(l2_norm / l2_norm_base * 100) + "%")
            f_BN.write(case_title + ", layer "+str(i) + " L2 norm: " + str(l2_norm) + "  ratio: " + str(l2_norm / l2_norm_base * 100) + "%\n")
            #if i <= 10:
            #    plot_distribution(data[s:t], output_folder, "Layer " + str(i) + " L2 distance distribution w.r.t. " + case_title)
            data_wanted = data_wanted + data[s:t].tolist()
            data_wanted_base = data_wanted_base + base_data[s:t].tolist()
        s = t
    
    print(len(data_wanted))
    l2_norm = np.linalg.norm(data_wanted)
    l2_norm_base = np.linalg.norm(data_wanted_base)
    value_list.append(l2_norm)
    value_percent_list.append(l2_norm / l2_norm_base * 100)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!" + case_title + ", total L2 norm: " + str(np.linalg.norm(data_wanted)) + "  ratio: " + str(l2_norm / l2_norm_base * 100) + "%")
    f_BN.write(case_title + ", total L2 norm: " + str(np.linalg.norm(data_wanted)) + "  ratio: " + str(l2_norm / l2_norm_base * 100) + "%\n")

    f_BN.write("\n")
    f_BN.write("\n")
    for value in value_list:
        f_BN.write("{:.3f} & ".format(value))
    f_BN.write("\n")
    for value_percent in value_percent_list:
        f_BN.write("{:.2f}\\% & ".format(value_percent))
    f_BN.write("\n")
    for value in value_list:
        f_BN.write("{:.3f}\t".format(value))
    f_BN.write("\n")
    for value_percent in value_percent_list:
        f_BN.write("{:.2f}\\%\t".format(value_percent))
    f_BN.write("\n")
    f_BN.close()
    plot_distribution(data, output_folder, "L2 distance distribution of all layers w.r.t. " + case_title)

def show_BN_statistics_difference():
    ROOT_DIR = os.path.abspath("../")
    print('Platform root: ', ROOT_DIR)
    DATA_DIR = ROOT_DIR + "/Data/udacityA_nvidiaB_results/test_results/"
    
    #BN_folder_1 = "(trainA)_(trainA)"
    BN_folder_1 = "(trainB)_(valB)"
    #BN_folder_1 = "(trainA)_(trainA)"
    #BN_folder_1 = "(trainA)_(valB)"
    #BN_folder_1 = "(trainC1)_(trainC1)"
    #BN_folder_1 = "(trainB_BN)_(valB)"
    #BN_folder_1 = "(trainB)_(trainA_advp)_(valB)"
    #BN_folder_1 = "(trainB_noise_3)_(valB_noise_3)"
    #BN_folder_2 = "(trainB)_(valB_distort_5)"
    #BN_folder_2 = "(trainB_noise_3)_(valB)"
    #BN_folder_2 = "(combine2)_(valB)"
    BN_folder_2 = "(trainB)_(trainA_similarBN10)"
    #BN_folder_2 = "(trainB)_(valA)"
    #BN_folder_2 = "(trainC1)_(valC1)"
    #BN_folder_2 = "(trainB_1_trainC1_BN_pretrain_reinitheader_BN1_reinitBN_BN)_(valB)"
    #BN_folder_2 = "(trainB)_(trainC1_advp)_(valB)"
    BN_flag = 0
    
    BN_means_1 = read_float_list(DATA_DIR + BN_folder_1 + "/BN_means.txt")
    BN_stds_1 = read_float_list(DATA_DIR + BN_folder_1 + "/BN_stds.txt")
    
    BN_means_2 = read_float_list(DATA_DIR + BN_folder_2 + "/BN_means.txt")
    BN_stds_2 = read_float_list(DATA_DIR + BN_folder_2 + "/BN_stds.txt")

    OUTPUT_DIR = ROOT_DIR + "/Data/udacityA_nvidiaB_results/BN_comparison/"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    output_folder = OUTPUT_DIR + "[" + BN_folder_1 + "][" + BN_folder_2 + "]/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    show_statistics(BN_means_1 - BN_means_2, BN_means_1, output_folder, case_title="mean", BN_flag=BN_flag)
    show_statistics(BN_means_1, BN_means_1, output_folder, case_title="mean_1", BN_flag=BN_flag)
    show_statistics(BN_means_2, BN_means_1, output_folder, case_title="mean_2", BN_flag=BN_flag)
    show_statistics(BN_stds_1 - BN_stds_2, BN_stds_1, output_folder, case_title="std", BN_flag=BN_flag)
    show_statistics(BN_stds_1, BN_stds_1, output_folder, case_title="std_1", BN_flag=BN_flag)
    show_statistics(BN_stds_2, BN_stds_1, output_folder, case_title="std_2", BN_flag=BN_flag)

def test_network_similarity():
    ROOT_DIR = os.path.abspath("../")
    print('Platform root: ', ROOT_DIR)
    TRAIN_OUTPUT_ROOT = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB_results/train_results")

    train_folder1 = "trainB_pytorch_newp"
    train_folder1 = "trainB_pytorch_newp1"
    modelPath1 = os.path.join(TRAIN_OUTPUT_ROOT, train_folder1, "model-final.pth")
    net1 = create_nvidia_network_pytorch(BN_flag=0)
    net1.load_state_dict(torch.load(modelPath1))

    param1 = []
    for param in net1.parameters():
        param1.append(param.detach().numpy())

    # train_folder2 = "trainB_pytorch_newp1"
    train_folder2 = "trainAds_pytorch_newp1"
    # train_folder2 = "trainB_pytorch_retrain_newp_reAds"
    modelPath2 = os.path.join(TRAIN_OUTPUT_ROOT, train_folder2, "model-final.pth")
    net2 = create_nvidia_network_pytorch(BN_flag=0)
    net2.load_state_dict(torch.load(modelPath2))

    param2 = []
    for param in net2.parameters():
        param2.append(param.detach().numpy())

    dist1 = []
    dist2 = []
    dist = []
    for i in range(len(param1)):
        dist1.append(np.linalg.norm(param1[i])/param1[i].size)
        dist2.append(np.linalg.norm(param2[i])/param1[i].size)
        dist.append(np.linalg.norm(param1[i]-param2[i]))

    print('')
    print(dist1)
    print('')
    print(dist2)
    print('')
    print(dist)
    print('')
    print(np.mean(dist1))
    print(np.mean(dist2))
    print(np.mean(dist))


if __name__ == "__main__":
    draw_label_distribution()
    #show_BN_statistics_difference()
    # test_network_similarity()