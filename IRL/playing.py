"""
Once a model is learned, use this to play it. 
It is running a policy to get its the feature expectations.
"""

from simulation import carmunk
import numpy as np
from neuralNets import net1
import sys
import time

NUM_FEATURES = 46 # number of features
NUM_ACTIONS = 25 # number of actions
GAMMA = 0.9


def play(model, weights, play_frames=10000, play_rounds=100):

    # init
    car_move = 0
    game_state = carmunk.GameState(weights, scene_file_name = 'scenes/scene-city.txt')
    _, state, _, _ = game_state.frame_step((2))
    featureExp = np.zeros(NUM_FEATURES)
    round_num = 0
    total_score = 0

    # start to move
    while True:
        car_move += 1

        # choose the best action
        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval))  

        # take the action
        reward , next_state, readings, score = game_state.frame_step(action)
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

        if readings[-1]==1:
            round_num += 1
            print("Score in this round: ", score)
            total_score += score
            print("Aver Score in ", round_num, "rounds: ", total_score / round_num)

        if round_num == play_rounds:
            break
        
        state = next_state

    return featureExp

if __name__ == "__main__": 
    #BEHAVIOR = sys.argv[1]
    #ITERATION = sys.argv[2]
    #FRAME = sys.argv[3]
    
    BEHAVIOR = "city"
    ITERATION = 20000
    FRAME = 1
    
    modelType = BEHAVIOR
    model_dir = 'results/models-'+ modelType +'/'
    saved_model = model_dir+'164-150-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'.h5'
    saved_weights = model_dir+'164-150-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'_weights.npy'
    weights = np.load(saved_weights)
    #weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
    model = net1(NUM_FEATURES, NUM_ACTIONS, [164, 150], saved_model)
    print (play(model, weights))
