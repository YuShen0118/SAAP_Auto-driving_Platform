#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
import os
import csv
import cv2

ROOT_DIR = os.path.abspath("../")
print('Platform root: ', ROOT_DIR)

dataset_path = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB/")
testres_path = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB_results/test_results/")

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


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


def get_image_name_list_from_csv(input_file, data_path):
    with open(input_file, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    image_name_list = []
    for row in trainLog:
        center_image = os.path.join(data_path, row[0])
        image_name_list.append(center_image)

    return image_name_list


def read_dataset(label_name, data_path, class_id):
    data_list = []
    class_id_list = []
    image_name_list = get_image_name_list_from_csv(label_name, data_path)
    for image_name in image_name_list:
        image = cv2.imread(image_name)

        image = cv2.resize(image,(64, 64), interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        data_list.append(image.flatten())
        class_id_list.append(class_id)

    return data_list, class_id_list


def read_steering_datasets(data_folders):
    # dataset_path = "C:/projects/SAAP_Auto-driving_Platform/Data/udacityA_nvidiaB/"                             # local
    # dataset_path = "/media/yushen/workspace2/projects/SAAP_Auto-driving_Platform/Data/udacityA_nvidiaB/"       # machine 3
    # dataset_path = "/scratch/yushen/SAAP_Auto-driving_Platform/Data/udacityA_nvidiaB/"                         # machine 17

    print('Reading steering datasets...')
    # data_folders = ['valBs', 'valHs', 'valAds', 'valAudi5', 'valAudi6']
    # data_folders = ['valBs', 'valHs', 'valAudi5', 'valAudi6']
    # data_folders = ['valBs', 'valHs']
    data_folders = ['trainAudi5', 'valAudi5', 'trainAudi6', 'valAudi6']

    data_list = []
    class_id_list = []
    for class_id, folder in enumerate(data_folders):
        print('Reading ', folder, '...')
        label_name = os.path.join(dataset_path, get_label_file_name(folder))
        data_path = os.path.join(dataset_path, folder)
        data_list_1, class_id_list_1 = read_dataset(label_name, data_path, class_id)
        data_list = data_list + data_list_1
        class_id_list = class_id_list + class_id_list_1

    print('Reading steering datasets finished!')
    return np.array(data_list) / 255.0, np.array(class_id_list)

def read_succ_fail_datasets(data_folders):
    print('Reading steering datasets (succ & fail)...')
    data_folders = ['trainAudi5', 'valAudi5']
    data_folders = ['trainAudi6', 'valAudi6']

    data_list = []
    class_id_list = []
    for class_id, folder in enumerate(data_folders):
        print('Reading ', folder, '...')
        label_name = os.path.join(dataset_path, get_label_file_name(folder))
        label_name = label_name.replace('.csv', '_1_succ.csv')
        data_path = os.path.join(dataset_path, folder)
        data_list_succ, class_id_list_succ = read_dataset(label_name, data_path, class_id*2)

        label_name = label_name.replace('_1_succ.csv', '_1_fail.csv')
        data_path = os.path.join(dataset_path, folder)
        data_list_fail, class_id_list_fail = read_dataset(label_name, data_path, class_id*2+1)

        data_list = data_list + data_list_succ + data_list_fail
        class_id_list = class_id_list + class_id_list_succ + class_id_list_fail

    print('Reading steering datasets finished!')
    return np.array(data_list) / 255.0, np.array(class_id_list)

def read_datasets_features(data_folders):
    net_name = "nvidiaNet5"
    net_name = "nvidiaNet6"
    net_name = "resnet5"
    net_name = "resnet6"
    net_name = "lstm5"
    net_name = "lstm6"

    net_name = "nvidiaNet"
    net_name = "resnet"
    # data_folders = ['trainAudi5', 'valAudi5', 'trainAudi6', 'valAudi6']

    train_folder = "trainB_pytorch"
    train_folder = "trainHc_all_rob_20epoch_single_BN8_pytorch"
    train_folder = "trainHc_BN8_pytorch"
    data_folders = ["valB", \
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

    data_folders = ["valB", \
                "valB_blur_5", \
                "valB_noise_5", \
                "valB_distort_5", \
                "valB_R_darker_5", \
                "valB_G_darker_5", \
                "valB_B_darker_5", \
                "valB_H_lighter_5", \
                "valB_S_lighter_5", \
                "valB_V_lighter_5", \
                "valB_combined_6_0", \
                "valB_IMGC_motion_blur_5", \
                "valB_IMGC_zoom_blur_5", \
                "valB_IMGC_pixelate_5", \
                "valB_IMGC_jpeg_compression_5", \
                "valB_IMGC_snow_5", \
                "valB_IMGC_frost_5", \
                "valB_IMGC_fog_5"
                ]

    data_folders = [
                "valB_blur_5", \
                "valB_V_lighter_5", \
                "valB_IMGC_snow_5", \
                "valB"
                ]

    data_list = []
    class_id_list = []
    for class_id, folder in enumerate(data_folders):

        if "trainHc" in train_folder:
            folder = folder.replace("valB", "valHc")

        # feature_file_name = os.path.join(dataset_path, folder, net_name + "_" + folder + "_feature.npy") # in the data folder
        feature_file_name = os.path.join(testres_path, "("+train_folder+")_("+folder+")", net_name + "_feature.npy")   # in the test folder
        feature = np.load(feature_file_name)
        class_id_list_1 = [class_id] * feature.shape[0]

        data_list = data_list + feature.tolist()
        class_id_list = class_id_list + class_id_list_1

    max_v = np.max(np.abs(np.array(data_list)))
    return np.array(data_list)/max_v, np.array(class_id_list)


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    # print("Running example on 2,500 MNIST digits...")
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")

    label_names = ['Nvidia', 'Honda', 'Audi_det', 'Audi_seg1', 'Audi_seg2']
    # label_names = ['valBs', 'valHs', 'valAds', 'valAudi5', 'valAudi6']
    label_names = ['trainAudi5', 'trainAudi6']
    label_names = ['Nvidia', 'Honda', 'Audi1', 'Audi2']
    label_names = ['Nvidia', 'Honda', 'Audi_seg1', 'Audi_seg2']
    label_names = ['valAudi5', 'valAudi6']
    label_names = ['Audi1_train', 'Audi1_val', 'Audi2_train', 'Audi2_val']


    label_names = ['Audi2_train_succ', 'Audi2_train_fail', 'Audi2_val_succ', 'Audi2_val_fail']

    label_names = ['Audi1_train', 'Audi1_test', 'Audi2_train', 'Audi2_test']

    label_names = ["Original", \
                "Blur_Lv5", \
                "Noise_Lv5", \
                "Distortion_Lv5", \
                "R_darker_Lv5", \
                "G_darker_Lv5", \
                "B_darker_Lv5", \
                "H_lighter_Lv5", \
                "S_lighter_Lv5", \
                "V_lighter_Lv5", \
                "Combination_No6", \
                "Motion_blur_Lv5", \
                "Zoom_blur_Lv5", \
                "Pixelate_Lv5", \
                "Jpeg_compression_Lv5", \
                "Snow_Lv5", \
                "Frost_Lv5", \
                "Fog_Lv5"
                ]

    label_names = [
                "Blur_Lv5", \
                "V_lighter_Lv5", \
                "Snow_Lv5", \
                "Original"
                ]

    # X, labels = read_steering_datasets(label_names)
    # X, labels = read_succ_fail_datasets(label_names)
    X, labels = read_datasets_features(label_names)

    # print(X)
    # print(labels)
    # adsf

    X = X[::2]
    labels = labels[::2]

    # Y = tsne(X, 2, 50, 20.0)
    Y = tsne(X, 2, 10, 20.0)
    for class_id in range(len(label_names)):
        clsss_mask = (labels==class_id)
        pylab.scatter(Y[:, 0][clsss_mask], Y[:, 1][clsss_mask], s=5, alpha=0.7, label=label_names[class_id])

    pylab.legend(loc="best")
    pylab.savefig('test.png', bbox_inches='tight')
    pylab.show()
