from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print('Platform root: ', ROOT_DIR)
#root = ROOT_DIR + '/Data/udacityA_nvidiaB/'
#print('Dataset root: ', root)

dataset_path = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB/")
#dataset_path = os.path.join("C:/projects/SAAP_Auto-driving_Platform/Data/nvidia/")
#dataset_path = os.path.join("/media/yushen/workspace/projects/SAAP_Auto-driving_Platform/Data/nvidia/")


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf

def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def get_cdf_of_images(image_paths, useHSV = True):
    all_hist_0 = []
    all_hist_1 = []
    all_hist_2 = []
    for image_path in image_paths:
        image = cv2.imread(image_path)

        if useHSV:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        c_0, c_1, c_2 = cv2.split(image)
        hist_0, _ = np.histogram(c_0.flatten(), 256, [0,256])
        hist_1, _ = np.histogram(c_1.flatten(), 256, [0,256])
        hist_2, _ = np.histogram(c_2.flatten(), 256, [0,256])
        
        all_hist_0.append(hist_0)
        all_hist_1.append(hist_1)
        all_hist_2.append(hist_2)
        
    cdf_0 = calculate_cdf(np.sum(all_hist_0, axis=0))
    cdf_1 = calculate_cdf(np.sum(all_hist_1, axis=0))
    cdf_2 = calculate_cdf(np.sum(all_hist_2, axis=0))

    return cdf_0, cdf_1, cdf_2

def get_lookup_table(cdfA, cdfB):
    # Make a separate lookup table for each color
    lookup_table_0 = calculate_lookup(cdfA[0], cdfB[0])
    lookup_table_1 = calculate_lookup(cdfA[1], cdfB[1])
    lookup_table_2 = calculate_lookup(cdfA[2], cdfB[2])

    return lookup_table_0, lookup_table_1, lookup_table_2


def get_cdfs(image_pathsA, image_pathsB, useHSV = True):
    cdfA = get_cdf_of_images(image_pathsA, useHSV)
    cdfB = get_cdf_of_images(image_pathsB, useHSV)

    return cdfA, cdfB

def transfer_image(src_image, lookup_table, useHSV = True):
    if useHSV:
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)

    src_0, src_1, src_2 = cv2.split(src_image)
    after_transform_0 = cv2.LUT(src_0, lookup_table[0])
    after_transform_1 = cv2.LUT(src_1, lookup_table[1])
    after_transform_2 = cv2.LUT(src_2, lookup_table[2])

    # Put the image back together
    image_after_matching = cv2.merge([after_transform_0, after_transform_1, after_transform_2])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)
    image_after_matching = cv2.cvtColor(image_after_matching, cv2.COLOR_HSV2BGR)

    return image_after_matching

def generate_dataset_transfer_color(folderA, folderB):
    useHSV = True
    #modeA = "/*.png" #GTA
    #modeB = "/*_leftImg8bit.png" #cityscapes
    modeA = "/center*.jpg" #udacity
    modeB = "/*.jpg" #nvidia
    image_pathsA = glob.glob(folderA + modeA)
    image_pathsB = glob.glob(folderB + modeB)
    cdfA, cdfB = get_cdfs(image_pathsA, image_pathsB, useHSV = useHSV)

    np.savetxt('cdfA_udacity.txt', cdfA)
    np.savetxt('cdfB_nvidia.txt', cdfB)

    print('cdfA_udacity.txt generated')
    print('cdfB_nvidia.txt generated')

    cdfA = np.loadtxt('cdfA_udacity.txt')
    cdfB = np.loadtxt('cdfB_nvidia.txt')
    
    lookup_table_A2B = get_lookup_table(cdfA, cdfB)
    lookup_table_B2A = get_lookup_table(cdfB, cdfA)

    folder_name = os.path.join(dataset_path, "trainA_fake_color")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    folder_name = os.path.join(dataset_path, "trainB_fake_color")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    folder_name = os.path.join(dataset_path, "valA_fake_color")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    folder_name = os.path.join(dataset_path, "valB_fake_color")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folderA = os.path.join(dataset_path, "trainA")
    image_pathsA = glob.glob(folderA + modeA)
    for image_path in image_pathsA:
        print(image_path)
        src_image = cv2.imread(image_path)
        new_image = transfer_image(src_image, lookup_table_A2B, useHSV = useHSV)
        new_image_path = image_path.replace('trainA', 'trainA_fake_color')

        cv2.imwrite(new_image_path, new_image)

        #style_image = cv2.imread(image_pathsB[0])
        #cv2.imshow('style_image', style_image)
    
        #cv2.imshow('src_image', src_image)
        #cv2.imshow('new_image', new_image)
        #cv2.waitKey(0)

    folderB = os.path.join(dataset_path, "trainB")
    image_pathsB = glob.glob(folderB + modeB)
    for image_path in image_pathsB:
        print(image_path)
        src_image = cv2.imread(image_path)
        new_image = transfer_image(src_image, lookup_table_B2A, useHSV = useHSV)
        new_image_path = image_path.replace('trainB', 'trainB_fake_color')

        cv2.imwrite(new_image_path, new_image)
        
    folderA = os.path.join(dataset_path, "valA")
    image_pathsA = glob.glob(folderA + modeA)
    for image_path in image_pathsA:
        print(image_path)
        src_image = cv2.imread(image_path)
        new_image = transfer_image(src_image, lookup_table_A2B, useHSV = useHSV)
        new_image_path = image_path.replace('valA', 'valA_fake_color')

        cv2.imwrite(new_image_path, new_image)
        
    folderB = os.path.join(dataset_path, "valB")
    image_pathsB = glob.glob(folderB + modeB)
    for image_path in image_pathsB:
        print(image_path)
        src_image = cv2.imread(image_path)
        new_image = transfer_image(src_image, lookup_table_B2A, useHSV = useHSV)
        new_image_path = image_path.replace('valB', 'valB_fake_color')

        cv2.imwrite(new_image_path, new_image)

def add_noise(image, sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def generate_dataset_diff_quality(dataset_path, folder):

    for level in range(5):
        blur_folder = dataset_path + folder + '_blur_'+str(level+1)
        if not os.path.exists(blur_folder):
            os.mkdir(blur_folder)
            
        noise_folder = dataset_path + folder + '_noise_'+str(level+1)
        if not os.path.exists(noise_folder):
            os.mkdir(noise_folder)

    #image_paths = glob.glob(dataset_path + folder + "/*_leftImg8bit.png")
    image_paths = glob.glob(dataset_path + folder + "/*.jpg")

    for image_path in image_paths:
        print(image_path)
        src_image = cv2.imread(image_path)

        '''
        #new_image = cv2.GaussianBlur(src_image, (15,15), 0)
        new_image = cv2.GaussianBlur(src_image, (7,7), 0)
        new_image_path = image_path.replace(folder, folder+'_blur_1')
        cv2.imwrite(new_image_path, new_image)

        #new_image = cv2.GaussianBlur(src_image, (35,35), 0)
        new_image = cv2.GaussianBlur(src_image, (17,17), 0)
        new_image_path = image_path.replace(folder, folder+'_blur_2')
        cv2.imwrite(new_image_path, new_image)

        #new_image = cv2.GaussianBlur(src_image, (75,75), 0)
        new_image = cv2.GaussianBlur(src_image, (37,37), 0)
        new_image_path = image_path.replace(folder, folder+'_blur_3')
        cv2.imwrite(new_image_path, new_image)

        new_image = cv2.GaussianBlur(src_image, (67,67), 0)
        new_image_path = image_path.replace(folder, folder+'_blur_4')
        cv2.imwrite(new_image_path, new_image)

        new_image = cv2.GaussianBlur(src_image, (107,107), 0)
        new_image_path = image_path.replace(folder, folder+'_blur_5')
        cv2.imwrite(new_image_path, new_image)
        '''


        '''
        new_image = add_noise(src_image, 20)
        new_image_path = image_path.replace(folder, folder+'_noise_1')
        cv2.imwrite(new_image_path, new_image)

        new_image = add_noise(src_image, 50)
        new_image_path = image_path.replace(folder, folder+'_noise_2')
        cv2.imwrite(new_image_path, new_image)

        new_image = add_noise(src_image, 100)
        new_image_path = image_path.replace(folder, folder+'_noise_3')
        cv2.imwrite(new_image_path, new_image)
        '''

        new_image = add_noise(src_image, 150)
        new_image_path = image_path.replace(folder, folder+'_noise_4')
        cv2.imwrite(new_image_path, new_image)

        new_image = add_noise(src_image, 200)
        new_image_path = image_path.replace(folder, folder+'_noise_5')
        cv2.imwrite(new_image_path, new_image)

    
def generate_dataset_distort(dataset_path, folder):
    for level in range(6):
        distort_folder = dataset_path + folder + '_distort_'+str(level+1)
        if not os.path.exists(distort_folder):
            os.mkdir(distort_folder)

    if 'label' in folder:
        image_paths = glob.glob(dataset_path + folder + "/*_gtFine_color.png")
    else:
        #image_paths = glob.glob(dataset_path + folder + "/*_leftImg8bit.png")
        image_paths = glob.glob(dataset_path + folder + "/*.jpg")

    for image_path in image_paths:
        print(image_path)
        src_image = cv2.imread(image_path)
        
        K = np.eye(3)*1000
        K[0,2] = src_image.shape[1]/2
        K[1,2] = src_image.shape[0]/2
        K[2,2] = 1

        '''
        #new_image = cv2.undistort(src_image, K, np.array([0.01,0.01,0,0]))
        new_image = cv2.undistort(src_image, K, np.array([0.1,0.1,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_1')
        cv2.imwrite(new_image_path, new_image)

        #new_image = cv2.undistort(src_image, K, np.array([0.1,0.1,0,0]))
        new_image = cv2.undistort(src_image, K, np.array([1,1,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_2')
        cv2.imwrite(new_image_path, new_image)

        #new_image = cv2.undistort(src_image, K, np.array([1,1,0,0]))
        new_image = cv2.undistort(src_image, K, np.array([10,10,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_3')
        cv2.imwrite(new_image_path, new_image)
        '''
        new_image = cv2.undistort(src_image, K, np.array([50,50,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_4')
        cv2.imwrite(new_image_path, new_image)

        new_image = cv2.undistort(src_image, K, np.array([200,200,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_5')
        cv2.imwrite(new_image_path, new_image)

        new_image = cv2.undistort(src_image, K, np.array([500,500,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_6')
        cv2.imwrite(new_image_path, new_image)


if __name__ == '__main__':
    #print(__doc__)
    folderA = os.path.join(dataset_path, "trainA")
    folderB = os.path.join(dataset_path, "trainB")

    #generate_dataset_transfer_color(folderA, folderB)

    
    #folder = "trainB_small"
    folder = "trainB"
    generate_dataset_diff_quality(dataset_path, folder)
    folder = "valB"
    generate_dataset_diff_quality(dataset_path, folder)

    
    folder = "trainB"
    generate_dataset_distort(dataset_path, folder)
    folder = "valB"
    generate_dataset_distort(dataset_path, folder)
    
    #folder = "labelsB"
    #generate_dataset_distort(dataset_path, folder)