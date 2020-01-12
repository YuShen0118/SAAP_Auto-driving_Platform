

import numpy as np
import cv2

from os import listdir
from os.path import isfile, join



def ProcessImage(img, count, output_path):
	img = cv2.imread(img)
	img = img[:,50:210,:] 
	cv2.imwrite(output_path+str(count)+'.png',img)


input_path = 'C:/DeepDrive/train/ICRA19/follow-curved/input/imgs/'
onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]

output_path = 'C:/Users/weizi/Downloads/virtual/'

count = 1
for i in onlyfiles:
	img_name = input_path + i
	ProcessImage(img_name, count, output_path)
	count += 1
	if count == 1100:
		break

	
'''
img = cv2.imread('C:/DeepDrive/train/ICRA19/follow-curved/input/imgs/center_2018_09_04_17_59_24_190.jpg')
img = img[:,50:210,:] 
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()	
'''




'''
def ProcessImage(img, count, output_path):
	img = cv2.imread(img)
	img = img[:,318:830,:] 
	small = cv2.resize(img, (0,0), fx=0.3125, fy=0.3125) 
	cv2.imwrite(output_path+str(count)+'.png',small)

input_path = 'C:/Users/weizi/Downloads/2011_09_26/2011_09_26_drive_0013_extract/image_03/data2/'
onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]

output_path = 'C:/Users/weizi/Downloads/real/'

count = 931
for i in onlyfiles:
	img_name = input_path + i
	ProcessImage(img_name, count, output_path)
	count += 1

img = cv2.imread('C:/Users/weizi/Downloads/2011_09_26/2011_09_26_drive_0013_extract/image_03/data2/0000000140.png')
img = img[:,318:830,:] 
small = cv2.resize(img, (0,0), fx=0.3125, fy=0.3125) 
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



print("hello world")
