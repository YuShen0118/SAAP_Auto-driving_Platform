from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print('Platform root: ', ROOT_DIR)
#root = ROOT_DIR + '/Data/udacityA_nvidiaB/'
#print('Dataset root: ', root)

dataset_path = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB/")
#dataset_path = os.path.join("C:/projects/SAAP_Auto-driving_Platform/Data/nvidia/")
#dataset_path = os.path.join("/media/yushen/workspace/projects/SAAP_Auto-driving_Platform/Data/nvidia/")

RGB_MAX = 255
HSV_H_MAX = 180
HSV_SV_MAX = 255
YUV_MAX = 255

# level values
BLUR_LVL = [7, 17, 37, 67, 107]
NOISE_LVL = [20, 50, 100, 150, 200]
DIST_LVL = [1, 10, 50, 200, 500]
RGB_LVL = [0.02, 0.2, 0.5, 0.65]

IMG_WIDTH = 200
IMG_HEIGHT = 66

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
    image_paths = glob.glob(os.path.join(dataset_path, folder, "*.png")) + glob.glob(os.path.join(dataset_path, folder, "*.jpg"))

    for image_path in image_paths:
        print(image_path)
        src_image = cv2.imread(image_path)

        
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
        

        new_image = add_noise(src_image, 20)
        new_image_path = image_path.replace(folder, folder+'_noise_1')
        cv2.imwrite(new_image_path, new_image)

        new_image = add_noise(src_image, 50)
        new_image_path = image_path.replace(folder, folder+'_noise_2')
        cv2.imwrite(new_image_path, new_image)

        new_image = add_noise(src_image, 100)
        new_image_path = image_path.replace(folder, folder+'_noise_3')
        cv2.imwrite(new_image_path, new_image)
        

        new_image = add_noise(src_image, 150)
        new_image_path = image_path.replace(folder, folder+'_noise_4')
        cv2.imwrite(new_image_path, new_image)

        new_image = add_noise(src_image, 200)
        new_image_path = image_path.replace(folder, folder+'_noise_5')
        cv2.imwrite(new_image_path, new_image)

    
def generate_dataset_distort(dataset_path, folder):
    for level in range(5):
        distort_folder = dataset_path + folder + '_distort_'+str(level+1)
        if not os.path.exists(distort_folder):
            os.mkdir(distort_folder)

    if 'label' in folder:
        image_paths = glob.glob(dataset_path + folder + "/*_gtFine_color.png")
    else:
        #image_paths = glob.glob(dataset_path + folder + "/*_leftImg8bit.png")
        image_paths = glob.glob(os.path.join(dataset_path, folder, "*.png")) + glob.glob(os.path.join(dataset_path, folder, "*.jpg"))

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
        '''

        #new_image = cv2.undistort(src_image, K, np.array([0.1,0.1,0,0]))
        new_image = cv2.undistort(src_image, K, np.array([1,1,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_1')
        cv2.imwrite(new_image_path, new_image)

        #new_image = cv2.undistort(src_image, K, np.array([1,1,0,0]))
        new_image = cv2.undistort(src_image, K, np.array([10,10,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_2')
        # added nov 10
        new_image = cv2.resize(new_image, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(new_image_path, new_image)
        
        new_image = cv2.undistort(src_image, K, np.array([50,50,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_3')
        # added nov 10
        new_image = cv2.resize(new_image, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(new_image_path, new_image)

        new_image = cv2.undistort(src_image, K, np.array([200,200,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_4')
        # added nov 10
        new_image = cv2.resize(new_image, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(new_image_path, new_image)

        new_image = cv2.undistort(src_image, K, np.array([500,500,0,0]))
        new_image_path = image_path.replace(folder, folder+'_distort_5')
        # added nov 10
        new_image = cv2.resize(new_image, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(new_image_path, new_image)


def get_edge_map(image):
    blur = cv2.GaussianBlur(image,(31,31),0)
    blur = blur / np.max(blur)
    blur = blur * 2 + image / np.max(image)
    edge_map = blur / np.max(blur)
    return edge_map


def transfer_to_edge_map(dataset_folder, out_folder):
    image_paths = glob.glob(dataset_folder + "/*.png")

    if len(image_paths) == 0:
        image_paths = glob.glob(dataset_folder + "/*.jpg")

    if len(image_paths) == 0:
        print("No images in " + dataset_folder)
        return

    for image_path in image_paths:
        print(image_path)
        image_ori = cv2.imread(image_path)
        #cv2.imshow("original image", image_ori)
        #edges = cv2.Canny(image_ori,100,200)
        laplacian = cv2.Laplacian(image_ori,cv2.CV_8U)

        #cv2.imshow("laplacian without border process between frames", laplacian)
        # deal with border between frames
        if image_ori.shape[0] == 128:
            laplacian[:,415:417,:] = np.zeros((laplacian.shape[0], 2, laplacian.shape[2]))
            laplacian[:,831:833,:] = np.zeros((laplacian.shape[0], 2, laplacian.shape[2]))

        '''
        #cv2.imshow("laplacian", laplacian)
        blur = cv2.GaussianBlur(laplacian,(31,31),0)
        #cv2.imshow("blurred laplacian", blur)
        blur = blur / np.max(blur)
        #cv2.imshow("rescaled blurred laplacian", blur)
        blur = blur + laplacian / np.max(laplacian)
        edge_map = blur / np.max(blur)
        #cv2.imshow("edge map", edge_map)
        #cv2.waitKey(0)
        '''
        edge_map = get_edge_map(laplacian)

        out_image_path = image_path.replace(os.path.dirname(image_path), out_folder+"/")
        cv2.imwrite(out_image_path, edge_map*255)
        #cv2.imwrite("test.jpg", laplacian)

    return



def transfer_to_combined_map(dataset_folder, out_folder):
    print("transfer_to_combined_map")
    image_paths = glob.glob(dataset_folder + "/*.png")

    if len(image_paths) == 0:
        image_paths = glob.glob(dataset_folder + "/*.jpg")

    if len(image_paths) == 0:
        print("No images in " + dataset_folder)
        return

    for image_path in image_paths:
        print(image_path)
        image_ori = cv2.imread(image_path, 0)
        #cv2.imshow("original image", image_ori)
        #edges = cv2.Canny(image_ori,100,200)
        laplacian = cv2.Laplacian(image_ori,cv2.CV_8U)

        #cv2.imshow("laplacian without border process between frames", laplacian)
        # deal with border between frames
        if image_ori.shape[0] == 128:
            laplacian[:,415:417] = np.zeros((laplacian.shape[0], 2))
            laplacian[:,831:833] = np.zeros((laplacian.shape[0], 2))

        laplacian_edge_map = get_edge_map(laplacian)

        canny_edges = cv2.Canny(image_ori,150,200)
        if image_ori.shape[0] == 128:
            canny_edges[:,415:417] = np.zeros((canny_edges.shape[0], 2))
            canny_edges[:,831:833] = np.zeros((canny_edges.shape[0], 2))

        canny_edge_map = get_edge_map(canny_edges)

        image_ori = image_ori / np.max(image_ori)

        combined_image = np.dstack((image_ori, laplacian_edge_map, canny_edge_map))

        '''
        cv2.imshow("image_ori", image_ori)
        cv2.imshow("laplacian_edge_map", laplacian_edge_map)
        cv2.imshow("canny_edge_map", canny_edge_map)
        cv2.imshow("combined_image", combined_image)
        cv2.waitKey(0)
        '''

        out_image_path = image_path.replace(os.path.dirname(image_path), out_folder+"/")
        cv2.imwrite(out_image_path, combined_image*255)

    return


def transfer_to_3_maps(dataset_path, dataset_folder, out_folders):
    dataset_folder = os.path.join(dataset_path, dataset_folder)

    print("transfer_to_3_maps")
    for out_folder in out_folders:
        out_folder = os.path.join(dataset_path, out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    image_paths = glob.glob(dataset_folder + "/*.png")

    if len(image_paths) == 0:
        image_paths = glob.glob(dataset_folder + "/*.jpg")

    if len(image_paths) == 0:
        print("No images in " + dataset_folder)
        return

    for image_path in image_paths:
        print(image_path)
        image_ori = cv2.imread(image_path, 0)
        #cv2.imshow("original image", image_ori)
        #edges = cv2.Canny(image_ori,100,200)
        laplacian = cv2.Laplacian(image_ori,cv2.CV_8U)

        #cv2.imshow("laplacian without border process between frames", laplacian)
        # deal with border between frames
        if image_ori.shape[0] == 128:
            laplacian[:,415:417] = np.zeros((laplacian.shape[0], 2))
            laplacian[:,831:833] = np.zeros((laplacian.shape[0], 2))

        laplacian_edge_map = get_edge_map(laplacian)

        canny_edges = cv2.Canny(image_ori,150,200)
        if image_ori.shape[0] == 128:
            canny_edges[:,415:417] = np.zeros((canny_edges.shape[0], 2))
            canny_edges[:,831:833] = np.zeros((canny_edges.shape[0], 2))

        canny_edge_map = get_edge_map(canny_edges)

        image_ori = image_ori / np.max(image_ori)

        combined_image = np.dstack((image_ori, laplacian_edge_map, canny_edge_map))

        '''
        cv2.imshow("image_ori", image_ori)
        cv2.imshow("laplacian_edge_map", laplacian_edge_map)
        cv2.imshow("canny_edge_map", canny_edge_map)
        cv2.imshow("combined_image", combined_image)
        cv2.waitKey(0)
        '''

        out_image_path0 = image_path.replace(os.path.dirname(image_path), os.path.join(dataset_path, out_folders[0]))
        cv2.imwrite(out_image_path0, laplacian)
        out_image_path1 = image_path.replace(os.path.dirname(image_path), os.path.join(dataset_path, out_folders[1]))
        cv2.imwrite(out_image_path1, canny_edges)
        out_image_path2 = image_path.replace(os.path.dirname(image_path), os.path.join(dataset_path, out_folders[2]))
        cv2.imwrite(out_image_path2, laplacian_edge_map*255)
        out_image_path3 = image_path.replace(os.path.dirname(image_path), os.path.join(dataset_path, out_folders[3]))
        cv2.imwrite(out_image_path3, canny_edge_map*255)
        out_image_path4 = image_path.replace(os.path.dirname(image_path), os.path.join(dataset_path, out_folders[4]))
        cv2.imwrite(out_image_path4, combined_image*255)

    return

def get_image_name_list_from_csv(input_file, originalDataset):
    with open(input_file, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    image_name_list = []
    for row in trainLog:
        center_image = os.path.join(originalDataset, row[0])
        image_name_list.append(center_image)

    return image_name_list

def list_image_files(dataset_folder):
    #image_paths = glob.glob(dataset_folder + "/*/*/image_02/data/*.png")
    image_paths = glob.glob(dataset_folder + "/image_02/data/*.png")
    file1 = open("myfile.txt","w")
    for image_path in image_paths:
        file1.write(image_path)
        file1.write('\n')
    file1.close()

def generate_RGB_dataset(originalDataset, channel, direction, dist_ratio=0.25, suffix='', csv_file=''):

    color_str_dic = {
        0: "B",
        1: "G", 
        2: "R"
    }
    
    color_str = color_str_dic.get(channel)
           
    direction_str = ""
    if direction == 0:
        direction_str = "darker"
    elif direction == 1:
        direction_str = "lighter"
    elif direction == 2:
        direction_str = "darker_5"
    elif direction == 3:
        direction_str = "lighter_5"
    elif direction == 4:
        direction_str = "darker_"+suffix
    elif direction == 5:
        direction_str = "lighter_"+suffix

    saveDir = "_".join([originalDataset, color_str, direction_str])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    image_name_list = glob.glob(os.path.join(originalDataset, "*.jpg")) + glob.glob(os.path.join(originalDataset, "*.png"))
    if csv_file != '':
        image_name_list = get_image_name_list_from_csv(csv_file, originalDataset)

    for i in image_name_list:
        # image_id = int(os.path.basename(i).replace('.jpg', ''))

        image = cv2.imread(str(i))
        
        if direction == 0: # lower the channel value
            image[:, :, channel] = image[:, :, channel] * 0.5
        elif direction == 1: # raise the channel value
            image[:, :, channel] = (image[:, :, channel] * 0.5) + (RGB_MAX * 0.5)
        elif direction == 2: # make channel value 0
            # print(image[:, :, channel])
            image[:, :, channel] = np.full(image[:, :, channel].shape, 0)
        elif direction == 3: # make channel value max value equal to 255
            image[:, :, channel] = np.full(image[:, :, channel].shape, RGB_MAX)
        elif direction == 4: # lower the channel value
            image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (0 * dist_ratio)
        elif direction == 5: # raise the channel value
            image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (RGB_MAX * dist_ratio)

        saveAsName = os.path.join(saveDir, os.path.basename(i))

        # added nov 10
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

        cv2.imwrite(saveAsName, image)

        # if image_id % 6000 == 0:
        #     print(saveAsName, ' generated')

def generate_HSV_datasets(originalDataset, channel, direction, dist_ratio=0.25, suffix='', csv_file=''):
    
    color_str_dic = {
        0: "H",
        1: "S", 
        2: "V"
    }
    
    color_str = color_str_dic.get(channel)
           
    direction_str = ""
    if direction == 0:
        direction_str = "darker"
    elif direction == 1:
        direction_str = "lighter"
    elif direction == 2:
        direction_str = "darker_5"
    elif direction == 3:
        direction_str = "lighter_5"
    elif direction == 4:
        direction_str = "darker_"+suffix
    elif direction == 5:
        direction_str = "lighter_"+suffix

    saveDir = "_".join([originalDataset, color_str, direction_str])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    max_val = HSV_SV_MAX
    if channel == 0:
        max_val = HSV_H_MAX
    
    image_name_list = glob.glob(os.path.join(originalDataset, "*.jpg")) + glob.glob(os.path.join(originalDataset, "*.png"))
    if csv_file != '':
        image_name_list = get_image_name_list_from_csv(csv_file, originalDataset)

    for i in image_name_list:
        # image_id = int(os.path.basename(i).replace('.jpg', ''))

        image_ori = cv2.imread(i)
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2HSV)
        
        if direction == 0:
            image[:, :, channel] = image[:, :, channel] * 0.5
        elif direction == 1:
            image[:, :, channel] = (image[:, :, channel] * 0.5) + (max_val * 0.5)
        elif direction == 2:
            if channel == 2:
                image[:, :, channel] = image[:, :, channel] * 0.1
            else:
                image[:, :, channel] = np.full(image[:, :, channel].shape, 0)
        elif direction == 3:
            image[:, :, channel] = np.full(image[:, :, channel].shape, max_val)
        elif direction == 4:
            image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio))
        elif direction == 5:
            image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (max_val * dist_ratio)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            
        saveAsName = os.path.join(saveDir, os.path.basename(i))
        
        # added nov 10
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(saveAsName, image)

        # if image_id % 6000 == 0:
        #     print(saveAsName, ' generated')

def generate_YUV_datasets(originalDataset, channel, direction, dist_ratio=0.25, suffix=''):
    
    color_str_dic = {
        0: "Y_luma",
        1: "U_blueproj", 
        2: "V_redproj"
    }
    
    color_str = color_str_dic.get(channel)
    
    direction_str = ""
    if direction == 0:
        direction_str = "darker"
    elif direction == 1:
        direction_str = "lighter"
    elif direction == 2:
        direction_str = "darker_2"
    elif direction == 3:
        direction_str = "lighter_2"
    elif direction == 4:
        direction_str = "darker_"+suffix
    elif direction == 5:
        direction_str = "lighter_"+suffix
    
    saveDir = "_".join([originalDataset, color_str, direction_str])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    img_list = glob.glob(os.path.join(originalDataset, "*.jpg")) + glob.glob(os.path.join(originalDataset, "*.png"))
    for i in img_list:
        temp = cv2.imread(i)
        image = temp.copy()

        # cv2.imshow( "Before", image )
        # cv2.waitKey(0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        if direction == 0:
            image[:, :, channel] = image[:, :, channel] * 0.5
        elif direction == 1:
            image[:, :, channel] = (image[:, :, channel] * 0.5) + (YUV_MAX * 0.5)
        elif direction == 2: # make channel value 0
            image[:, :, channel] = np.full(image[:, :, channel].shape, 0)
        elif direction == 3: # make channel value max value == 255
            image[:, :, channel] = np.full(image[:, :, channel].shape, YUV_MAX)
        elif direction == 4: # lower the channel value
            image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (0 * dist_ratio)
        elif direction == 5: # raise the channel value
            image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (YUV_MAX * dist_ratio)

        # image[:, :, 0] = image[:, :, 0]*0
        # image[:, :, 1] = image[:, :, 1]*0
        # image[:, :, 2] = image[:, :, 2]*0

        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
            
        saveAsName = os.path.join(saveDir, os.path.basename(i))
        # cv2.imshow( "After", image )
        # cv2.waitKey(0)
        # break # comment
        cv2.imwrite(saveAsName, image)

KSIZE_MIN = 0.1
KSIZE_MAX = 3.8
NOISE_MIN = 0.1
NOISE_MAX = 4.6
DISTORT_MIN = -2.30258509299
DISTORT_MAX = 5.3
COLOR_SCALE = 0.25

# generate the combined parameters, the max values for each parameter for a generation of one combination
def get_combined_parameters():
    alpha = np.zeros(6)
    gaussian_ksize = 0
    noise_level = 0
    distort_level = 0

    alpha = np.random.normal(loc=0,scale=0.6,size=6)

    gaussian_ksize = int(np.exp(np.random.uniform(KSIZE_MIN, KSIZE_MAX, 1))[0])

    if gaussian_ksize % 2 == 0: # kernel size must be even
        gaussian_ksize += 1

    noise_level = int(np.exp(np.random.uniform(NOISE_MIN, NOISE_MAX, 1))[0])
    distort_level = int(np.random.uniform(0.1, 50, 1)[0])

    return alpha, gaussian_ksize, noise_level, distort_level #distort_level

def read_parameters_from_file(parameter_file):
    with open(parameter_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            alpha = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            gaussian_ksize = int(float(row[6]))
            noise_level = int(float(row[7]))
            distort_level = int(float(row[8]))
        # print(alpha)
    return alpha, gaussian_ksize, noise_level, distort_level

# samples factors over a uniform distribution
# factors: blur, noise, distortion, R, G, B, H, S, V
def generate_combined(originalDataset, id, parameter_file='', csv_file='', dist_ratio=0.25, numDatasets=1):
    # alpha = np.zeros(6)
    # gaussian_ksize = 0
    # noise_level = 0
    # distort_level = 0
    dist_ranges = [1, 3, 10, 20, 40, 70]
    noise_ranges = [2, 5, 8, 13, 20, 30]
    blur_ranges = [2, 8, 15, 35, 50, 67]
    alpha_ranges = [0.1, 0.2, 0.5, 0.65, 0.8, 1]

    # these are actually the max values
    for j in range(numDatasets):

        if parameter_file == '':
            alpha2, gaussian2, noise2, dist2 = get_combined_parameters()
        else:
            alpha2, gaussian2, noise2, dist2 = read_parameters_from_file(parameter_file)

    
        saveDir = "_".join([originalDataset, "combined", str(id), str(j)])
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        # print(alpha)
        # alpha2 = alpha / alpha_ranges[5] * alpha_ranges[l]
        # gaussian2 = int(gaussian_ksize / blur_ranges[5] * blur_ranges[l])
        # if gaussian2 % 2 == 0: # kernel size must be even
        #     gaussian2 += 1
        
        # noise2 = int(noise_level / noise_ranges[5] * noise_ranges[l])
        # gaussian2 = int(np.random.uniform(blur_ranges[l], blur_ranges[l+1]))
        # if gaussian2 % 2 == 0: # kernel size must be even
        #     gaussian2 += 1
        # noise2 = int(np.random.uniform(noise_ranges[l], noise_ranges[l+1]))
        # dist2 = int(np.random.uniform(dist_ranges[l], dist_ranges[l+1]))
        # dist2 = int(distort_level / dist_ranges[5] * dist_ranges[l])

        # print(alpha, gaussian_ksize, noise_level, distort_level)
        # # sample parameters
        # alpha = np.random.normal(loc=0,scale=0.25,size=6)
        # # gaussian_ksize = np.random.randint(3, high=11, size=1)[0]
        # gaussian_ksize = int(np.exp(np.random.uniform(0.1, 4.20469261939, 1))[0])
        
        # if gaussian_ksize % 2 == 0: # kernel size must be even
        #     gaussian_ksize += 1

        # # noise_level = np.random.randint(0, high=200, size=1)[0]
        # # noise_level = abs(int(np.random.normal(loc=0,scale=17,size=1)[0]))
        # noise_level = int(np.exp(np.random.uniform(0.1, 3.91202300543, 1))[0])
        # # distort_level = np.random.randint(0, high=50, size=1)[0]

        # # log-uniform sampling, low = ln(0.1) < -2, and high = ln(500) < 7
        # distort_level = int(np.exp(np.random.uniform(-2.30258509299, 6.21460809842, 1))[0])

        # write parameters to parameters.txt in saveDir
        f = open(os.path.join(saveDir,"parameters.txt"),"w+")
        parameters_concat = np.concatenate((alpha2, [gaussian2], [noise2], [dist2]))
        parameters_concat = [str(s) for s in parameters_concat]
        write_str = ','.join(parameters_concat)
        f.write("B,G,R,H,S,V,blur_ksize,noise_level,distort_level\n")
        f.write(write_str + "\n")
        f.close()

        image_name_list = glob.glob(os.path.join(originalDataset, "*.jpg")) + glob.glob(os.path.join(originalDataset, "*.png"))
        print(originalDataset)
        if csv_file != '':
            image_name_list = get_image_name_list_from_csv(csv_file, originalDataset)
        
        for i in image_name_list:
            # image_id = int(os.path.basename(i).replace('.jpg', ''))
            img = cv2.imread(i).copy()
            
            # adding color channel distortion on RGB, HSV, 6 channels total

            for channel in range(3):
                if alpha2[channel] < 0:
                    img[:, :, channel] = (img[:, :, channel] * (1+alpha2[channel]))
                else:
                    img[:, :, channel] = (img[:, :, channel] * (1-alpha2[channel])) + (RGB_MAX * alpha2[channel])
            
            for channel in range(3):
                max_val = 255
                if channel == 0:
                    max_val = 180
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if alpha2[channel+3] < 0:
                    img[:, :, channel] = (img[:, :, channel] * (1+alpha2[channel+3]))
                else:
                    img[:, :, channel] = (img[:, :, channel] * (1-alpha2[channel+3])) + (max_val * alpha2[channel+3])
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            # adding blur
            img = cv2.GaussianBlur(img, (gaussian2,gaussian2), 0)
            
            # adding noise
            img = add_noise(img, noise2)

            # adding distortion
            K = np.eye(3)*1000
            K[0,2] = img.shape[1]/2
            K[1,2] = img.shape[0]/2
            K[2,2] = 1

            img = cv2.undistort(img, K, np.array([dist2, dist2, 0, 0]))

            saveAsName = os.path.join(saveDir, os.path.basename(i))
        
            cv2.imwrite(saveAsName, img)

            # if image_id % 6000 == 0:
            #     print(saveAsName, ' generated')
            
        
        print('generated at ', saveDir)

# lvl can be 1, 2, 3, 4
# factor can be "blur", "noise", "distort", "R", "G", "B", "H", "S", "V"
def _generate_middle_level(originalDataset, lvl, lo, hi, factor, direction=0, csv_file=''):
    
    if factor in ["blur", "noise", "distort"]:
        saveDir = "_".join([originalDataset, factor, str(lvl) + "-5"])
    else:
        if direction == 0:
            saveDir = "_".join([originalDataset, factor, "darker", str(lvl) + "-5"])
        elif direction == 1:
            saveDir = "_".join([originalDataset, factor, "lighter", str(lvl) + "-5"])

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    image_name_list = glob.glob(os.path.join(originalDataset, "*.jpg")) + glob.glob(os.path.join(originalDataset, "*.png"))
    print(originalDataset)
    if csv_file != '':
        image_name_list = get_image_name_list_from_csv(csv_file, originalDataset)
    
    for i in image_name_list:
        # image_id = int(os.path.basename(i).replace('.jpg', ''))
        img = cv2.imread(i).copy()

        new_val = int((hi+lo)/2)

        if factor == "blur":
            if new_val % 2 == 0:
                new_val += 1

            img = cv2.GaussianBlur(img, (new_val,new_val), 0)
        elif factor == "noise":
            img = add_noise(img, new_val)
        elif factor == "distort":
            K = np.eye(3)*1000
            K[0,2] = img.shape[1]/2
            K[1,2] = img.shape[0]/2
            K[2,2] = 1
            img = cv2.undistort(img, K, np.array([new_val, new_val, 0, 0]))

        elif factor in ["R", "G", "B", "H", "S", "V"]:
            if factor in ["B", "H"]:
                channel = 0
            elif factor in ["G", "S"]:
                channel = 1
            else:
                channel = 2

            if lvl == 1:
                alpha = 0.08
            elif lvl == 2:
                alpha = 0.3
            elif lvl == 3:
                alpha = 0.58
            elif lvl == 4:
                alpha = 0.8
            
            if factor in ["H", "S", "V"]:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            if direction == 0:
                img[:, :, channel] = (img[:, :, channel] * (1-alpha))
            else:
                if factor == "H":
                    img[:, :, channel] = (img[:, :, channel] * (1-alpha)) + (HSV_H_MAX * alpha)
                else:
                    img[:, :, channel] = (img[:, :, channel] * (1-alpha)) + (RGB_MAX * alpha)
            
            if factor in ["H", "S", "V"]:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        saveAsName = os.path.join(saveDir, os.path.basename(i))
    
        cv2.imwrite(saveAsName, img)
        # if image_id % 6000 == 0:
        #     print(saveAsName, ' generated')


def generate_middle_blur(originalDataset):
    _generate_middle_level(originalDataset, 1, BLUR_LVL[0], BLUR_LVL[1], "blur")
    _generate_middle_level(originalDataset, 2, BLUR_LVL[1], BLUR_LVL[2], "blur")
    _generate_middle_level(originalDataset, 3, BLUR_LVL[2], BLUR_LVL[3], "blur")
    _generate_middle_level(originalDataset, 4, BLUR_LVL[3], BLUR_LVL[4], "blur")

def generate_middle_noise(originalDataset):
    _generate_middle_level(originalDataset, 1, NOISE_LVL[0], NOISE_LVL[1], "noise")
    _generate_middle_level(originalDataset, 2, NOISE_LVL[1], NOISE_LVL[2], "noise")
    _generate_middle_level(originalDataset, 3, NOISE_LVL[2], NOISE_LVL[3], "noise")
    _generate_middle_level(originalDataset, 4, NOISE_LVL[3], NOISE_LVL[4], "noise")

def generate_middle_distort(originalDataset):
    _generate_middle_level(originalDataset, 1, DIST_LVL[0], DIST_LVL[1], "distort")
    _generate_middle_level(originalDataset, 2, DIST_LVL[1], DIST_LVL[2], "distort")
    _generate_middle_level(originalDataset, 3, DIST_LVL[2], DIST_LVL[3], "distort")
    _generate_middle_level(originalDataset, 4, DIST_LVL[3], DIST_LVL[4], "distort")

def generate_middle_colors(originalDataset):
    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "R", direction = 0)
    _generate_middle_level(originalDataset, 1, 0.2, 0.5, "R", direction = 1)
    _generate_middle_level(originalDataset, 2, 0.02, 0.2, "R", direction = 0)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "R", direction = 1)
    _generate_middle_level(originalDataset, 3, 0.02, 0.2, "R", direction = 0)
    _generate_middle_level(originalDataset, 3, 0.2, 0.5, "R", direction = 1)
    _generate_middle_level(originalDataset, 4, 0.02, 0.2, "R", direction = 0)
    _generate_middle_level(originalDataset, 4, 0.2, 0.5, "R", direction = 1)

    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "G", direction = 0)
    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "G", direction = 1)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "G", direction = 0)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "G", direction = 1)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "G", direction = 0)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "G", direction = 1)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "G", direction = 0)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "G", direction = 1)

    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "B", direction = 0)
    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "B", direction = 1)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "B", direction = 0)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "B", direction = 1)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "B", direction = 0)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "B", direction = 1)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "B", direction = 0)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "B", direction = 1)

    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "H", direction = 0)
    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "H", direction = 1)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "H", direction = 0)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "H", direction = 1)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "H", direction = 0)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "H", direction = 1)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "H", direction = 0)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "H", direction = 1)

    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "S", direction = 0)
    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "S", direction = 1)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "S", direction = 0)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "S", direction = 1)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "S", direction = 0)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "S", direction = 1)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "S", direction = 0)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "S", direction = 1)

    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "V", direction = 0)
    _generate_middle_level(originalDataset, 1, 0.02, 0.2, "V", direction = 1)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "V", direction = 0)
    _generate_middle_level(originalDataset, 2, 0.2, 0.5, "V", direction = 1)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "V", direction = 0)
    _generate_middle_level(originalDataset, 3, 0.5, 0.65, "V", direction = 1)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "V", direction = 0)
    _generate_middle_level(originalDataset, 4, 0.65, 1, "V", direction = 1)

def generate_all_middle(originalDataset):
    print('generating datasets for ' + originalDataset)
    generate_middle_blur(originalDataset)
    print('finished generating blur middle datasets')
    generate_middle_noise(originalDataset)
    print('finished generating noise middle datasets')
    generate_middle_distort(originalDataset)
    print('finished generating distort middle datasets')
    # generate_middle_colors(originalDataset)
    # print('finished generating color middle datasets')

def generate_all_color_levels(dataFolder, dataFolderVal, csvFile, csvFileVal):
    # train folder
    generate_RGB_dataset(dataFolder, 0, 4, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 0, 5, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 4, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 5, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 4, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 5, dist_ratio=0.02, suffix='1', csv_file=csvFile)

    generate_HSV_datasets(dataFolder, 0, 4, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 0, 5, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 4, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 5, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 4, dist_ratio=0.02, suffix='1', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 5, dist_ratio=0.02, suffix='1', csv_file=csvFile)

    generate_RGB_dataset(dataFolder, 0, 4, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 0, 5, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 4, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 5, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 4, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 5, dist_ratio=0.2, suffix='2', csv_file=csvFile)

    generate_HSV_datasets(dataFolder, 0, 4, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 0, 5, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 4, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 5, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 4, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 5, dist_ratio=0.2, suffix='2', csv_file=csvFile)
    
    generate_RGB_dataset(dataFolder, 0, 4, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 0, 5, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 4, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 5, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 4, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 5, dist_ratio=0.5, suffix='3', csv_file=csvFile)

    generate_HSV_datasets(dataFolder, 0, 4, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 0, 5, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 4, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 5, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 4, dist_ratio=0.5, suffix='3', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 5, dist_ratio=0.5, suffix='3', csv_file=csvFile)

    generate_RGB_dataset(dataFolder, 0, 4, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 0, 5, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 4, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 5, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 4, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 5, dist_ratio=0.65, suffix='4', csv_file=csvFile)

    generate_HSV_datasets(dataFolder, 0, 4, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 0, 5, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 4, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 5, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 4, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 5, dist_ratio=0.65, suffix='4', csv_file=csvFile)
    
    generate_RGB_dataset(dataFolder, 0, 4, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 0, 5, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 4, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 1, 5, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 4, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_RGB_dataset(dataFolder, 2, 5, dist_ratio=1, suffix='5', csv_file=csvFile)

    generate_HSV_datasets(dataFolder, 0, 4, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 0, 5, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 4, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 1, 5, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 4, dist_ratio=1, suffix='5', csv_file=csvFile)
    generate_HSV_datasets(dataFolder, 2, 5, dist_ratio=1, suffix='5', csv_file=csvFile)
    
    # validation folder
    generate_RGB_dataset(dataFolderVal, 0, 4, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 0, 5, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 4, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 5, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 4, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 5, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)

    generate_HSV_datasets(dataFolderVal, 0, 4, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 0, 5, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 4, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 5, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 4, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 5, dist_ratio=0.02, suffix='1', csv_file=csvFileVal)
    
    generate_RGB_dataset(dataFolderVal, 0, 4, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 0, 5, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 4, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 5, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 4, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 5, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)

    generate_HSV_datasets(dataFolderVal, 0, 4, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 0, 5, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 4, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 5, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 4, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 5, dist_ratio=0.2, suffix='2', csv_file=csvFileVal)
    
    generate_RGB_dataset(dataFolderVal, 0, 4, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 0, 5, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 4, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 5, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 4, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 5, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)

    generate_HSV_datasets(dataFolderVal, 0, 4, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 0, 5, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 4, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 5, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 4, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 5, dist_ratio=0.5, suffix='3', csv_file=csvFileVal)

    generate_RGB_dataset(dataFolderVal, 0, 4, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 0, 5, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 4, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 5, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 4, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 5, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)

    generate_HSV_datasets(dataFolderVal, 0, 4, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 0, 5, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 4, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 5, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 4, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 5, dist_ratio=0.65, suffix='4', csv_file=csvFileVal)

    generate_RGB_dataset(dataFolderVal, 0, 4, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 0, 5, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 4, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 1, 5, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 4, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_RGB_dataset(dataFolderVal, 2, 5, dist_ratio=1, suffix='5', csv_file=csvFileVal)

    generate_HSV_datasets(dataFolderVal, 0, 4, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 0, 5, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 4, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 1, 5, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 4, dist_ratio=1, suffix='5', csv_file=csvFileVal)
    generate_HSV_datasets(dataFolderVal, 2, 5, dist_ratio=1, suffix='5', csv_file=csvFileVal)

    print('finished generating all color datasets')

def generate_all(code):
    dataFolder = os.path.join(dataset_path, "train" + code)
    csvFile = os.path.join(dataset_path, "labels" + code + "_train.csv")
    dataFolderVal = os.path.join(dataset_path, "val" + code)
    csvFileVal = os.path.join(dataset_path, "labels" + code + "_val.csv")

    generate_all_color_levels(dataFolder, dataFolderVal, csvFile, csvFileVal)
    generate_dataset_diff_quality(dataset_path, "train" + code)
    print("finished generating noise and blur for train" + code)
    generate_dataset_diff_quality(dataset_path, "val" + code)
    print("finished generating noise and blue for val" + code)
    generate_dataset_distort(dataset_path, "train" + code)
    print("finished generating distortion for train" + code)
    generate_dataset_distort(dataset_path, "val" + code)
    print("finished generating distortion for val" + code)

    generate_combined(dataFolderVal, "1", parameter_file=os.path.join(dataset_path, "valB_combined_3_0_LVL5", "parameters.txt"))
    generate_combined(dataFolderVal, "2", parameter_file=os.path.join(dataset_path, "valB_combined_4_0_LVL5", "parameters.txt"))
    generate_combined(dataFolderVal, "3", parameter_file=os.path.join(dataset_path, "valB_combined_7_0_LVL5", "parameters.txt"))
    generate_combined(dataFolderVal, "4", parameter_file=os.path.join(dataset_path, "valB_combined_8_0_LVL5", "parameters.txt"))
    generate_combined(dataFolderVal, "5", parameter_file=os.path.join(dataset_path, "valB_combined_9_0_LVL5", "parameters.txt"))
    generate_combined(dataFolderVal, "6", parameter_file=os.path.join(dataset_path, "valB_combined_10_0_LVL5", "parameters.txt"))

if __name__ == '__main__':
    #print(__doc__)
    '''
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
    '''

    '''
    folder = "valB"
    transfer_to_3_maps(dataset_path, folder, [folder+"_lap", folder+"_canny", folder+"_lap_blur", folder+"_canny_blur", folder+"_comb"])
    folder = "valA"
    transfer_to_3_maps(dataset_path, folder, [folder+"_lap", folder+"_canny", folder+"_lap_blur", folder+"_canny_blur", folder+"_comb"])
    folder = "valC1"
    transfer_to_3_maps(dataset_path, folder, [folder+"_lap", folder+"_canny", folder+"_lap_blur", folder+"_canny_blur", folder+"_comb"])
    folder = "trainB"
    transfer_to_3_maps(dataset_path, folder, [folder+"_lap", folder+"_canny", folder+"_lap_blur", folder+"_canny_blur", folder+"_comb"])
    folder = "trainA"
    transfer_to_3_maps(dataset_path, folder, [folder+"_lap", folder+"_canny", folder+"_lap_blur", folder+"_canny_blur", folder+"_comb"])
    folder = "trainC1"
    transfer_to_3_maps(dataset_path, folder, [folder+"_lap", folder+"_canny", folder+"_lap_blur", folder+"_canny_blur", folder+"_comb"])
    '''
    
    generate_all("Hc")
    # generate_all("Ads")