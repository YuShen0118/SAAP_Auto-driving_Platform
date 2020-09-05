### This script is for preprocessing utility functions.

import os
import sys
import csv
import math
import random
import cv2    
import numpy as np
import matplotlib.pyplot as plt



####################################################
####################################################
##      Image Proceessing
####################################################
####################################################
def display_CV2(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(img, fDrive = False):
	'''
    Resize an image from 160x320x3 (i.e. Unity resolution) to 66x200x3 (i.e. nVidia paper's resolution)
	This method is used both for training and driving with the following difference.
    Driving:  RGB2YUV (because we process an image from Unity which is in RGB)
    Training: BGR2YUV (because we use cv2 to read an image which is in BGR)
    '''
    ## crop off the bottom portion (i.e. car hood) and the top portion (i.e. sky)
	#newImg = img[20:130,:,:] 
	newImg = img
	## apply the subtle blur
	fBlur = False
	if fBlur:
		newImg = cv2.GaussianBlur(newImg, (3,3), 0)
		
	newImg = cv2.resize(newImg,(200, 66), interpolation = cv2.INTER_AREA)
	if fDrive:
		newImg = cv2.cvtColor(newImg, cv2.COLOR_RGB2YUV)
	else:
		newImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2YUV)
	
	#cv2.imshow('img',img)
	#cv2.imshow('newImg',newImg)
	#cv2.waitKey(1)

	return newImg
	
	
def process_img_for_visualization(image, angle, anglePredicted, frameIdx):
    '''
    Used in visualize_dataset method to format image prior to displaying. 
    Converts colorspace back to original BGR, applies text to display steering angle and frame number (within batch to be visualized), 
    and applies lines representing steering angle and model-predicted steering angle (if available) to image.
    '''    
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    ## apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=0.5, color=(200,100,100), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=0.5, color=(200,100,100), thickness=1)
    ## apply a line representing the steering angle
    cv2.line(img, (int(w/2),int(h)), (int(w/2+angle*w/4), int(h/2)), (0,255,0), thickness=4)
    if anglePredicted is not None:
        cv2.line(img, (int(w/2),int(h)), (int(w/2+anglePredicted*w/4), int(h/2)), (0,0,255), thickness=4)
    return img
    
    
def visualize_dataset(imageList, angleList, anglePredictedList = None):
    '''
    Format the data to display
    '''
    for i in range(len(imageList)):
        if anglePredictedList is not None:
            img = process_img_for_visualization(imageList[i], angleList[i], anglePredictedList[i], i)
        else: 
            img = process_img_for_visualization(imageList[i], angleList[i], None, i)
        display_CV2(img) 


def visualize_train_data(xList, yList, batchSize, randomDistortFlag = False):
    '''
    Generate a batch of training data for visualization 
    '''
    X,y = ([],[])
    for i in range(batchSize):
        img = resize_image(cv2.imread(xList[i]))
        angle = yList[i]
        if randomDistortFlag:
            img, angle = random_distort(img, angle)
        X.append(img)
        y.append(angle)
        display_CV2(img)
    return (np.array(X), np.array(y))
    

def random_distort(image, angle):
    ''' 
    Method for adding random distortion to images, including brightness adjust, shadow, and vertical shift
    '''
    new_img = image.astype(float)
    ## random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    ## random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    ## random vertical shift
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8, h/8)
    pts1 = np.float32([[0,horizon], [w,horizon], [0,h], [w,h]])
    pts2 = np.float32([[0,horizon+v_shift], [w,horizon+v_shift], [0,h], [w,h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(new_img, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)
    
    
    
####################################################
####################################################
##      Preprocessing
####################################################
####################################################
def load_train_data(xFolder, trainLogPath, nRep, fThreeCameras = False):
	'''
	Load the training data
	'''
	## prepare for getting x
	if not os.path.exists(xFolder):
		sys.exit('Error: the image folder is missing. ' + xFolder)
		
	## prepare for getting y
	if not os.path.exists(trainLogPath):
		sys.exit('Error: the labels.csv is missing. ' + trainLogPath)
	with open(trainLogPath, newline='') as f:
		trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
	
    ## get x and y
	xList, yList = ([], [])
	
	for rep in range(0,nRep):
		for row in trainLog:  
			## center camera
	 		xList.append(xFolder + os.path.basename(row[0])) 
 			yList.append(float(row[3]))     
 			
 			## if using three cameras
 			if fThreeCameras:

				## left camera
 				xList.append(xFolder + row[1])  
 				yList.append(float(row[3]) + 0.25) 
				
				## right camera
 				xList.append(xFolder + row[2])  
 				yList.append(float(row[3]) - 0.25) 
			
	#yList = np.array(yList)*10 + 10
	return (xList, yList)

def load_train_data_multi(xFolder_list, trainLogPath_list, nRep, fThreeCameras = False, ratio = 1.0, specialFilter = False):
	'''
	Load the training data
	'''
	## prepare for getting x
	for xFolder in xFolder_list:
		if not os.path.exists(xFolder):
			sys.exit('Error: the image folder is missing. ' + xFolder)
		
	## prepare for getting y
	trainLog_list = []
	for trainLogPath in trainLogPath_list:
		if not os.path.exists(trainLogPath):
			sys.exit('Error: the labels.csv is missing. ' + trainLogPath)
		with open(trainLogPath, newline='') as f:
			trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
			trainLog_list.append(trainLog)

	if not isinstance(ratio, list):
		ratio = [ratio]*len(xFolder_list)
	
    ## get x and y
	xList, yList = ([], [])
	
	for rep in range(0,nRep):
		i = 0
		for trainLog in trainLog_list:
			xFolder = xFolder_list[i]
			xList_1 = []
			yList_1 = []
			for row in trainLog:
				## center camera
				if not specialFilter:
					xList_1.append(xFolder + os.path.basename(row[0])) 
					yList_1.append(float(row[3]))     
				elif float(row[3]) < 0:
					xList_1.append(xFolder + os.path.basename(row[0])) 
					yList_1.append(float(row[3]))
	 			
	 			## if using three cameras
				if fThreeCameras:

					## left camera
					xList_1.append(xFolder + row[1])  
					yList_1.append(float(row[3]) + 0.25) 
					
					## right camera
					xList_1.append(xFolder + row[2])  
					yList_1.append(float(row[3]) - 0.25) 

			if ratio[i] < 1:
				n = int(len(trainLog) * ratio[i])
				random.seed(42)
				random.shuffle(xList_1)
				random.seed(42)
				random.shuffle(yList_1)
				xList_1 = xList_1[0:n]
				yList_1 = yList_1[0:n]
			print(len(xList_1))
			xList = xList + xList_1
			yList = yList + yList_1

			i+=1

	#yList = np.array(yList)*10 + 10
	return (xList, yList)
	
def load_train_data_aux(trainFolder, imageList, auxList, angleList):
	'''
    Load the training data, append to imageList, auxList and angleList, and optionally repeat them 
    '''
	with open(trainFolder + 'log.csv', newline='') as f:
		trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
	with open(trainFolder + 'zone-log', newline='') as f:
		zoneLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
	imageFolder = trainFolder + 'imgs/'
	for row in trainLog:
		imageList.append(imageFolder + row[0].split('/')[-1])  # center camera image
		angleList.append(float(row[3])) # steering angle
	for row in zoneLog:
		auxList.append(int(row[0]))
	
	return (imageList, auxList, angleList)
	
	
def balance_train_data(imageList, angleList):
	'''
    Balance the training data, make them more equally distributed
    '''
	## show the histogram of the current data
	binwidth = 0.005
	plt.hist(angleList, bins=np.arange(min(angleList), max(angleList) + binwidth, binwidth), rwidth=0.8)
	plt.title('Number of images per steering angle')
	plt.xlabel('Steering Angle')
	plt.ylabel('# Frames')
	plt.show()

	## print a histogram to see which steering angle ranges are most overrepresented
	numBins = 23
	avgSamplesPerBin = len(angleList)/numBins
	hist, bins = np.histogram(angleList, numBins)
	width = 0.7 * (bins[1] - bins[0]) # decide each bar's width
	center = (bins[:-1] + bins[1:]) / 2 # decide the bar graph center
	plt.bar(center, hist, align='center', width=width) # plot all bars
	plt.plot((np.min(angleList), np.max(angleList)), (avgSamplesPerBin, avgSamplesPerBin), 'k-') # plot the average sample black line
	plt.show()
	
	## determine keep probability for each bin: if below avgSamplesPerBin, keep all; otherwise keep prob is proportional
	## to number of samples above the average, so as to bring the number of samples for that bin down to the average
	keepProbs = []
	target = avgSamplesPerBin * 0.5
	for i in range(numBins):
		if hist[i] < target:
			keepProbs.append(1.)
		else:
			keepProbs.append(1./(hist[i]/target))
	removeList = []
	for i in range(len(angleList)):
		for j in range(numBins):
			if angleList[i] > bins[j] and angleList[i] <= bins[j+1]:
				## delete from X and y with probability 1 - keep_probs[j]
				if np.random.rand() > keepProbs[j]:
					removeList.append(i)
	imageList = np.delete(imageList, removeList, axis=0)
	angleList = np.delete(angleList, removeList)

	## print histogram again to show more even distribution of steering angles
	hist, bins = np.histogram(angleList, numBins)
	plt.bar(center, hist, align='center', width=width)
	plt.plot((np.min(angleList), np.max(angleList)), (avgSamplesPerBin, avgSamplesPerBin), 'k-')
	plt.show()
	print('After:', imageList.shape, angleList.shape)
	
	

if __name__ == "__main__":
	print('\n')
	print("### This is the library file for the preprocessing process. Please do not directly run it.")
	print('\n')


'''
# read a csv file on Ubuntu
with open(input-file, 'rb') as f:
    data = list(csv.reader(f))
'''

'''
# Use GPU 2 of GMS for now
os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
'''	
