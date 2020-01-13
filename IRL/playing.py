"""
Once a model is learned, use this to play it. 
It is running a policy to get its the feature expectations.
"""

from simulation import carmunk
import numpy as np
from neuralNets import net1
import sys
import time

NUM_FEATURES = 8 # number of features
NUM_ACTIONS = 3 # number of actions
GAMMA = 0.9


def play(model, weights, play_frames):

    # init
    car_move = 0
    game_state = carmunk.GameState(weights)
    _, state, __ = game_state.frame_step((2))
    featureExp = np.zeros(NUM_FEATURES)

    # start to move
    while True:
        car_move += 1

        # choose the best action
        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval))  

        # take the action
        reward , next_state, readings = game_state.frame_step(action)
        #print ("reward: ", reward)
        #print ("readings: ", readings)

        # start recording feature expectations only after 100 frames
        if car_move > 100:
            featureExp += (GAMMA**(car_move-101))*np.array(readings)
        #print ("featureExp: ", featureExp)

        # Tell us something.
        if car_move % play_frames == 0:
            print("The car has moved %d frames" % car_move)
            break

    return featureExp

if __name__ == "__main__": 
    BEHAVIOR = sys.argv[1]
    ITERATION = sys.argv[2]
    FRAME = sys.argv[3]
    
    modelType = BEHAVIOR
    model_dir = 'results_models-'+ modelType +'/'
    saved_model = model_dir+'-164-150-100-50000-'+str(FRAME)+str(ITERATION)+'.h5'
    weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
    model = neural_net(NUM_INPUT, NUM_OUTPUT, [164, 150], saved_model)
    print (play(model, weights))
