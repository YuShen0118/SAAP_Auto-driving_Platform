"""
Once a model is learned, use this to play it. 
It is running a policy to get its the feature expectations.
"""

from simulation import carmunk
import numpy as np
from neuralNets import net1
import sys
import time
import timeit
import random

NUM_FEATURES = 46 # number of features
NUM_ACTIONS = 25 # number of actions
GAMMA = 0.99


def play(model, weights, play_frames=10000, play_rounds=10000, scene_file_name='scenes/scene-city.txt'):
    return play_multi_model([model], [1], weights, play_frames, play_rounds, scene_file_name)


def play_multi_model(model_list, lamda_list, weights, play_frames=10000, play_rounds=10000, scene_file_name='scenes/scene-city.txt'):

    # init
    car_move = 0
    game_state = carmunk.GameState(weights, scene_file_name = scene_file_name)
    _, state, _, _, _ = game_state.frame_step((11))
    featureExp = np.zeros(NUM_FEATURES)
    round_num = 0
    score_list = []
    dist_list = []
    dist_1round = 0
    step_1round = 0
    max_step_1round = 3000

    time_list = []

    # start to move
    while True:
        start_time = timeit.default_timer()
        car_move += 1
        step_1round += 1

        # choose the best action
        randv = random.uniform(0, 1)
        model_id = -1
        while randv>=0 and model_id<len(lamda_list)-1:
            model_id += 1
            randv -= lamda_list[model_id]

        model_id = np.clip(model_id, 0, len(lamda_list) - 1)

        model = model_list[model_id]

        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval)) 

        #TODO
        #action = random.randrange(0, len(qval.flatten()))

        # take the action
        reward , next_state, readings, score, dist_1step = game_state.frame_step(action)
        dist_1round += dist_1step
        #print ("reward: ", reward)
        #print ("readings: ", readings)

        # start recording feature expectations only after 100 frames
        if car_move > 100:
            featureExp += (GAMMA**(car_move-101))*np.array(readings)
        #print ("featureExp: ", featureExp)

        # Tell us something.
        if readings[-1]==1 or step_1round==max_step_1round:
            step_1round = 0
            round_num += 1
            score_list.append(score)
            dist_list.append(dist_1round)
            #print("Score in this round: ", score)
            #print("Aver Score in ", round_num, "rounds: ", np.average(score_list))
            #print("Dist in this round: ", dist_1round)
            #print("Aver dist in ", round_num, "rounds: ", np.average(dist_list))
            dist_1round = 0
            game_state.reinit_car()

        if play_frames > 0 and car_move % play_frames == 0:
            #print("The car has moved %d frames" % car_move)
            if readings[-1] == 0:
                round_num += 1
                score_list.append(score)
                dist_list.append(dist_1round)
            #print("Score in this round: ", score)
            #print("Aver Score in ", round_num, "rounds: ", np.average(score_list))
            #print("Dist in this round: ", dist_1round)
            #print("Aver dist in ", round_num, "rounds: ", np.average(dist_list))
            break

        if play_rounds > 0 and round_num == play_rounds:
            #print("Score in this round: ", score)
            #print("Aver Score in ", round_num, "rounds: ", np.average(score_list))
            #print("Dist in this round: ", dist_1round)
            #print("Aver dist in ", round_num, "rounds: ", np.average(dist_list))
            break
        
        state = next_state
        time_list.append(timeit.default_timer() - start_time)
        
        #print("fps: ", 1 / np.average(time_list), " ")

    print("min score=", np.min(score_list))
    print("max score=", np.max(score_list))
    print("aver score=", np.average(score_list))
    print("standard deviation score=", np.std(score_list))
    print("min dist=", np.min(dist_list))
    print("max dist=", np.max(dist_list))
    print("aver dist=", np.average(dist_list))
    print("standard deviation dist=", np.std(dist_list))

    return featureExp, np.average(score_list), np.average(dist_list)

if __name__ == "__main__": 
    #BEHAVIOR = sys.argv[1]
    #ITERATION = sys.argv[2]
    #FRAME = sys.argv[3]
    
    BEHAVIOR = "city"
    ITERATION = 20000
    FRAME = 1
    score_list = []
    dist_list = []
    
    for FRAME in range(1,3):
        print('***************************************************************************************************')
        print('FRAME ', FRAME)
        modelType = BEHAVIOR
        #model_dir = 'results/models-'+ modelType +'/'
        model_dir = 'results/finals/'
        model_dir = 'results/models-city/'
        saved_model = model_dir+'164-150-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'.h5'
        saved_model = model_dir+'64-128-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'.h5'
        weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
        # model = net1(NUM_FEATURES, NUM_ACTIONS, [164, 150], saved_model)
        model = net1(NUM_FEATURES, NUM_ACTIONS, [64, 128, 64, 32, 16, 8, 16, 32, 64, 128, 64], saved_model)
        
        scene_file_name = 'scenes/scene-city-car.txt'
        scene_file_name = 'scenes/scene-ground-car.txt'
        scene_file_name = 'scenes/scene-city.txt'
        reward_weights=''
        featureExp, score, dist = play(model, weights=reward_weights, play_rounds=100, scene_file_name = scene_file_name)
        score_list.append(score)
        dist_list.append(dist)

        for feature in featureExp:
            print('{:.3f}'.format(feature), end =", ")

    print('***************************************************************************************************')
    for i in range(len(score_list)):
        print(i+1, 'score', score_list[i], 'dist', dist_list[i])


