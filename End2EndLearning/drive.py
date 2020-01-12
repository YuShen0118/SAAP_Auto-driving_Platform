### This script is to load a model and use it to drive an AV in the simulator

from keras import __version__ as keras_version
from keras.models import load_model
import h5py
import argparse
import base64
import os
import shutil
import cv2
import csv
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from flask import Flask
from io import BytesIO
from datetime import datetime

import sys
sys.path.insert(0, 'library/')
from utilities import resize_image


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired_speed(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement
        # integral error
        self.integral = self.integral + self.error
        return self.Kp * self.error + self.Ki * self.integral



###################################
## variables
desiredSpeed = 20

## LSTM
fLSTM = False
fOnlyFollow = True
frameCount = 0
nFramesLSTM = 5
curSampleLSTM = np.empty((nFramesLSTM, 66, 200, 3))

# !!!! hard coded path
netPath =  'D:/data/Kitti/object/training_simu_20200109/trainedModels/models-cnn/'
netModel = netPath + 'model130.h5' 

###################################

## globals
controller = SimplePIController(0.1, 0.002)
controller.set_desired_speed(desiredSpeed) 
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def record_images():
	if args.image_folder != '':
		timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
		image_filename = os.path.join(args.image_folder, timestamp)
		image.save('{}.jpg'.format(image_filename))


def shift_img_array(curSampleLSTM, newImg):
	global nFramesLSTM
	newSampleLSTM = curSampleLSTM
	
	for i in range(nFramesLSTM - 1):
		newSampleLSTM[i] = curSampleLSTM[i+1]
		
	newSampleLSTM[-1] = newImg
	return newSampleLSTM
	

@sio.on('telemetry')
def telemetry(sid, data):
	global frameCount
	global curSampleLSTM
	global nFramesLSTM
	if data:
		## get data from Unity
		angleUnity = data["steering_angle"]
		throttleUnity = data["throttle"]
		speedUnity = data["speed"]
		ctrImgUnity = resize_image(np.array(Image.open(BytesIO(base64.b64decode(data["image"])))), True)
		

		## prepare variables for prediction results
		rAngle = 0
		rThrottle = controller.update(float(speedUnity))	
		
		## if LSTM is used, prepare a LSTM sample for predicting
		if not fLSTM:
			ctrImgModel = ctrImgUnity[None, :, :, :]
			#rDetect = np.argmax(netDetect.predict(ctrImgModel))
			rAngle = float(driveModel.predict(ctrImgModel))
		else:
			if frameCount < nFramesLSTM:
				curSampleLSTM[frameCount] = ctrImgUnity
				frameCount += 1
			else:
				curSamplePredict = curSampleLSTM[None, :, :, :, :] 
				rAngle = np.mean(driveModel.predict(curSamplePredict)) 
				#rAngleTest = np.mean(netTest.predict(curSamplePredict))
				curSampleLSTM = shift_img_array(curSampleLSTM, ctrImgUnity)
			
				
		print('speedUnity ' + str(speedUnity))
		print('rAngle ' + str(rAngle))
		print('rThrottle ' + str(rThrottle))
		send_control(rAngle*float(speedUnity)/150, rThrottle)
		
	else:
		sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

	## take arguments from the command line
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument(
		'model',
		type=str,
		help='Path to model h5 file. Model should be on the same path.'
	)

	'''
	parser.add_argument(
		'image_folder',
		type=str,
		nargs='?',
		default='',
		help='Path to image folder. This is where the images from the run will be saved.'
	)
	'''


	#args = parser.parse_args()

	'''
	if args.image_folder != '':
		print("Creating image folder at {}".format(args.image_folder))
		if not os.path.exists(args.image_folder):
			os.makedirs(args.image_folder)
		else:
			shutil.rmtree(args.image_folder)
			os.makedirs(args.image_folder)
		print("An image storing folder is provided, recording this run ...")
	else:
		print("An image storing folder is missing, not recording this run ...")
		
	'''
	## check that the model's Keras version is the same as local Keras version
	
	#f = h5py.File(netPath + args.model, mode='r')
	f = h5py.File(netModel, mode='r')
	model_version = f.attrs.get('keras_version')
	keras_version = str(keras_version).encode('utf8')
	if model_version != keras_version:
		print('\n')
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
		print('Current keras version ', keras_version, ', model keras version ', model_version)
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
		print('\n')

    
	## load a model
	#driveModel = load_model(netPath + args.model)
	driveModel = load_model(netModel)

	## wrap Flask application with engineio's middleware
	app = socketio.Middleware(sio, app)

	## deploy as an eventlet WSGI server
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


