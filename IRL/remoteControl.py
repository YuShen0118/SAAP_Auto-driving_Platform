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
import math

import sys
sys.path.insert(0, 'library/')

sio = socketio.Server()
app = Flask(__name__)


game_state = carmunk.GameState(scene_file_name = 'scenes/scene-city.txt', use_expert=True)

frameIdx = 0

@sio.on('telemetry')
def telemetry(sid, data):
    global frameCount
    global curSampleLSTM
    global nFramesLSTM
    global frameIdx
    if data:
        frameIdx += 1
        print("------------------frameIdx ", frameIdx, "-----------------------")
        ## get data from Unity
        angleUnity = data["steering_angle"]
        throttleUnity = data["throttle"]
        speedUnity = data["speed"]
        carPos = [float(data["mainCar_position_x"]), float(data["mainCar_position_y"])]
        print(carPos)
        carVelo = [float(data["mainCar_velocity_x"]), float(data["mainCar_velocity_y"])]
        carAngle = float(data["mainCar_direction"])

        # change coordinate
        carAngle = math.pi / 2 - carAngle / 180 * math.pi
        
        action = game_state.get_expert_action_out(carPos, carVelo, carAngle)
        [steer_angle, acceleration] = game_state.get_instruction_from_action_out(action)

        maxAngle = math.pi / 2
        steer_angle = -steer_angle / maxAngle
        print("steer_angle ", steer_angle)

        #game_state.draw()
        game_state.frame_step(action, effect=False)

        print("acceleration ", acceleration)
        send_control(steer_angle, acceleration)
        
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

    ## wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    ## deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


