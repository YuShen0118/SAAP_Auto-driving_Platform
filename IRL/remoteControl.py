### This script is to load a model and use it to drive an AV in the simulator

from keras import __version__ as keras_version
from keras.models import load_model
import h5py
import argparse
import base64
import os
import shutil
import csv
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from io import BytesIO
from datetime import datetime
from flask import Flask
from simulation import carmunk
from neuralNets import net1
import math

import sys
sys.path.insert(0, 'library/')

sio = socketio.Server()
app = Flask(__name__)


#game_state = carmunk.GameState(scene_file_name = 'scenes/scene-city.txt', use_expert=True)

frameIdx = 0

NUM_FEATURES = 46 # number of features
NUM_ACTIONS = 25 # number of actions
saved_model = 'results/finals/164-150-100-50000-20000-3.h5'
model = net1(NUM_FEATURES, NUM_ACTIONS, [164, 150], saved_model)

#game_state = carmunk.GameState(scene_file_name = 'scenes/scene-city.txt')
game_state = carmunk.GameState(scene_file_name = 'scenes/scene-city-car.txt')
_, state, _, _, _ = game_state.frame_step((11))

@sio.on('telemetry')
def telemetry(sid, data):
    global frameCount
    global curSampleLSTM
    global nFramesLSTM
    global frameIdx
    global model
    global game_state
    global state
        
    if data:
        frameIdx += 1
        print("------------------frameIdx ", frameIdx, "-----------------------")
        ## get data from Unity
        angleUnity = data["steering_angle"]
        throttleUnity = data["throttle"]
        speedUnity = data["speed"]
        carPos = [float(data["mainCar_position_x"]), float(data["mainCar_position_y"])]
        carVelo = [float(data["mainCar_velocity_x"]), float(data["mainCar_velocity_y"])]
        carAngle = float(data["mainCar_direction"])

        
        #action = game_state.get_expert_action_out(carPos, carVelo, carAngle)
        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval))  

        [steer_angle, acceleration] = game_state.get_instruction_from_action_out(action)

        reward , next_state, readings, score, dist_1step = game_state.frame_step(action)
        
        [carPosOut, carVeloOut, carAngleOut] = game_state.get_car_info()

        [car1_pos, car2_pos] = game_state.get_other_car_info()
        
        #goal_position = [16, -63]
        goal_position = [-3.5, 38]
        delta = np.array(goal_position) - np.array(carPosOut)
        if (np.linalg.norm(delta) < 2):
            exit()

        maxAngle = math.pi / 2
        steer_angle = -steer_angle / maxAngle

        send_control(steer_angle, acceleration, carPosOut, carVeloOut, carAngleOut, car1_pos, car2_pos)
        
        state = next_state
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    [carPosOut, carVeloOut, carAngleOut] = game_state.get_car_info()
    [car1_pos, car2_pos] = game_state.get_other_car_info()
    send_control(0, 0, carPosOut, carVeloOut, carAngleOut, car1_pos, car2_pos)


def send_control(steering_angle, throttle, carPosOut, carVeloOut, carAngleOut, car1_pos, car2_pos):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'mainCar_position_x': carPosOut[0].__str__(),
            'mainCar_position_y': carPosOut[1].__str__(),
            'mainCar_velocity_x': carVeloOut[0].__str__(),
            'mainCar_velocity_y': carVeloOut[1].__str__(),
            'mainCar_direction': carAngleOut.__str__(),
            
            'otherCar1_position_x': car1_pos[0].__str__(),
            'otherCar1_position_y': car1_pos[1].__str__(),
            'otherCar2_position_x': car2_pos[0].__str__(),
            'otherCar2_position_y': car2_pos[1].__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':

    ## wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    ## deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


