''' Standardizes label format to be readable by end 2 end model, from canonical downloaded format.  
    Desired format: [center img pth],[left img path],[right img path],[steering angle]
    
    NVIDIA original format: [img path] [steering angle],[YYYY-MM-DD] [HH:MM:SS:MS]

    Udacity original format: [center ABSOLUTE img pth],[left img path],[right img path],[steering angle],[throttle],[reverse],[speed]
        
    Custom format: [center img path],[left img path],[right img path],[steering angle]
'''
import csv, os, sys
from pathlib import Path

#### CHANGE THESE AS NEEDED ####
NVIDIA = True
Udacity = True

root = 'C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/'

# NVIDIA data folder
NVIDIA_path = root + 'NVIDIA/'

# Udacity data folder
UDACITY_path1 = root + 'Udacity/track1data/'
UDACITY_path2 = root + 'Udacity/track2data/'

# original label file locations
NVIDIA_original_label_file = NVIDIA_path + 'data.txt'

UDACITY_original_label_file1 = UDACITY_path1 + 'driving_log.csv' # track 1
UDACITY_original_label_file2 = UDACITY_path2 + 'driving_log.csv' # track 2

#### CHANGE THESE AS NEEDED ####



def convert_NVIDIA():
    ''' Converts NVIDIA format to custom format readable by end 2 end model.
        Writes to a csv file named "formatted_labels.csv" in NVIDIA data directory.
        Original format:
            [img path] [steering angle],[YYYY-MM-DD] [HH:MM:SS:MS]
        Resulting format:
            [img pth],,,[steering angle]
    '''
    
    if not os.path.exists(NVIDIA_original_label_file):
        sys.exit('Error: the NVIDIA labels file is missing.')
    
    with open(NVIDIA_original_label_file, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    f = open(NVIDIA_path + 'formatted_labels.csv', "w")
    
    for row in trainLog:
        image = row[0].split()[0]
        angle = row[0].split()[1]
        
        output = ','.join([image,"","",angle])
        
        # print(output)
        f.write(output + '\n')
    f.close()
    

def convert_UDACITY():
    ''' Converts UDACITY format to custom format readable by end 2 end model.
        Writes to a csv file named "formatted_labels.csv" in Udacity data directory.
        Original format:
            [center ABSOLUTE img pth],[left ABSOLUTE img path],[right ABSOLUTE img path],[steering angle],[throttle],[reverse],[speed]
        Resulting format:
            [center RELATIVE img path],[left RELATIVE img path],[right RELATIVE img path],[steering angle]
    '''
    # track 1
    if not os.path.exists(UDACITY_original_label_file1):
        sys.exit('Error: the UDACITY labels file is missing.')
    
    with open(UDACITY_original_label_file1, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    f = open(UDACITY_path1 + 'formatted_labels.csv', "w")
    
    for row in trainLog:
        center_image = Path(row[0]).name
        left_image = Path(row[1]).name
        right_image = Path(row[2]).name
        angle = row[3]
        
        output = ','.join([center_image,left_image,right_image,angle])
        
        f.write(output + '\n')
    f.close()
    
    # track 2
    if not os.path.exists(UDACITY_original_label_file2):
        sys.exit('Error: the UDACITY labels file is missing.')
    
    with open(UDACITY_original_label_file2, newline='') as f:
        trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    f = open(UDACITY_path2 + 'formatted_labels.csv', "w")
    
    for row in trainLog:
        center_image = Path(row[0]).name
        left_image = Path(row[1]).name
        right_image = Path(row[2]).name
        angle = row[3]
        
        output = ','.join([center_image,left_image,right_image,angle])
        
        f.write(output + '\n')
    f.close()

    










if NVIDIA:
    convert_NVIDIA()

if Udacity:
    convert_UDACITY()
    
    

