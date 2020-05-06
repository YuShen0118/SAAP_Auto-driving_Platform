''' Standardizes label format to be readable by end 2 end model, from canonical downloaded format.  
    Desired format: [center img pth],[left img path],[right img path],[steering angle]
    
    NVIDIA original format: [img path] [steering angle],[YYYY-MM-DD] [HH:MM:SS:MS]

    Udacity original format: [center ABSOLUTE img pth],[left img path],[right img path],[steering angle],[throttle],[reverse],[speed]
        
    Custom format: [center img path],[left img path],[right img path],[steering angle]
'''
import csv, os, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT_DIR = os.path.abspath("../")
print('Platform root: ', ROOT_DIR)

#### CHANGE THESE AS NEEDED ####
NVIDIA = True
Udacity = True
SPLIT = True

root = ROOT_DIR + '/Data/udacityA_nvidiaB/'
print('Dataset root: ', root)

# NVIDIA dataset from Sully Chen records the actual angle of the steering wheel
# To get the angle the actual wheels turn, divide the recorded angles by the ratio
# Recorded on a 2014 Honda Civic: https://hondanews.com/en-US/honda-automobiles/releases/release-b228c382366b432890e04499ebb2b995-2014-civic-si-specifications-and-features
NVIDIA_STEERING_RATIO = 15.06

#### CHANGE THESE AS NEEDED ####

# needed to scale steering angle in Udacity data back to actual angle measurements
UDACITY_MAX_STEERING_ANGLE = 25

def convert_NVIDIA(input_file, output_file):
    ''' Converts NVIDIA format to custom format readable by end 2 end model.
        Writes to a csv file named "formatted_labels.csv" in NVIDIA data directory.
        Original format:
            [img path] [steering angle],[YYYY-MM-DD] [HH:MM:SS:MS]
        Resulting format:
            [img pth],,,[steering angle]
    '''
    
    if not os.path.exists(input_file):
        sys.exit('Error: the NVIDIA labels file is missing.')
    
    with open(input_file, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    f = open(output_file, "w")
    
    for row in trainLog:
        image = row[0].split()[0]
        angle = str(float(row[0].split()[1]) / NVIDIA_STEERING_RATIO)
        
        output = ','.join([image,"","",angle])
        
        # print(output)
        f.write(output + '\n')
    f.close()
    

def convert_UDACITY(input_file, output_file):
    ''' Converts UDACITY format to custom format readable by end 2 end model.
        Writes to a csv file named "formatted_labels.csv" in Udacity data directory.
        Original format:
            [center ABSOLUTE img pth],[left ABSOLUTE img path],[right ABSOLUTE img path],[steering angle],[throttle],[reverse],[speed]
        Resulting format:
            [center RELATIVE img path],[left RELATIVE img path],[right RELATIVE img path],[steering angle]
    '''
    if not os.path.exists(input_file):
        sys.exit('Error: the UDACITY labels file is missing.')
    
    with open(input_file, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    f = open(output_file, "w")

    angles = []
    for row in trainLog:
        angle = float(row[3]) * UDACITY_MAX_STEERING_ANGLE
        angles.append(angle)

    #angles_filtered = gaussian_filter1d(angles, 1)
    angles_filtered = angles
    
    i=0
    for row in trainLog:
        center_image = Path(row[0]).name
        left_image = Path(row[1]).name
        right_image = Path(row[2]).name
        #angle = float(row[3]) * UDACITY_MAX_STEERING_ANGLE
        angle = str(angles_filtered[i])
        
        output = ','.join([center_image,left_image,right_image,str(angle)])
        
        f.write(output + '\n')
        i+=1
    f.close()
    
#split the trainval labels into train labels and val labels, according to the images in the train folder
def split(imgs_train_folder, lables_trainval_filename, lables_train_filename, lables_val_filename, step_size=1):
    with open(lables_trainval_filename, newline='') as f:
        labels_trainval = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    train_mask = []
    for row in labels_trainval:
        center_image = Path(row[0]).name
        if os.path.exists(imgs_train_folder + center_image):
            train_mask.append(True)
        else:
            train_mask.append(False)

    train_mask = np.array(train_mask)
    labels_trainval = np.array(labels_trainval)
    labels_train = labels_trainval[train_mask]
    val_mask = ~train_mask
    labels_val = labels_trainval[val_mask]

    labels_train = labels_train[0::step_size]
    labels_val = labels_val[0::step_size]

    lables_train_filename = lables_trainval_filename.replace('trainval', 'train')
    f = open(lables_train_filename, "w")
    for row in labels_train:
        center_image = Path(row[0]).name
        left_image = Path(row[1]).name
        right_image = Path(row[2]).name
        angle = row[3]
        
        output = ','.join([center_image,left_image,right_image,angle])
        
        f.write(output + '\n')
    f.close()

    lables_val_filename = lables_trainval_filename.replace('trainval', 'val')
    f = open(lables_val_filename, "w")
    for row in labels_val:
        center_image = Path(row[0]).name
        left_image = Path(row[1]).name
        right_image = Path(row[2]).name
        angle = row[3]
        
        output = ','.join([center_image,left_image,right_image,angle])
        
        f.write(output + '\n')
    f.close()

if Udacity:
    input_file = root + "labelsA_ori.csv"
    output_file = root + "labelsA_trainval.csv"
    convert_UDACITY(input_file, output_file)
    print("convert udacity labels successfully!")

if NVIDIA:
    input_file = root + "labelsB_ori.txt"
    output_file = root + "labelsB_trainval.csv"
    convert_NVIDIA(input_file, output_file)
    print("convert nvidia labels successfully!")

    
if SPLIT:
    imgs_train_folder = root + 'trainA/'
    lables_trainval_filename = root + 'labelsA_trainval.csv'
    lables_train_filename = root + 'labelsA_train.csv'
    lables_val_filename = root + 'labelsA_val.csv'
    split(imgs_train_folder, lables_trainval_filename, lables_train_filename, lables_val_filename)
    print("split udacity labels successfully!")

    imgs_train_folder = root + 'trainB/'
    lables_trainval_filename = root + 'labelsB_trainval.csv'
    lables_train_filename = root + 'labelsB_train.csv'
    lables_val_filename = root + 'labelsB_val.csv'
    split(imgs_train_folder, lables_trainval_filename, lables_train_filename, lables_val_filename, step_size=6)
    print("split nvidia labels successfully!")
