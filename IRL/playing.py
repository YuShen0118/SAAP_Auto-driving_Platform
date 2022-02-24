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
import os.path

# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd

NUM_FEATURES = 46 # number of features
NUM_ACTIONS = 25 # number of actions
# NUM_ACTIONS = 5 # number of actions
GAMMA = 0.99


def play(model, weights, play_frames=10000, play_rounds=10000, scene_file_name='scenes/scene-city.txt', reward_net=None, use_expert=False, return_path=False):
    return play_multi_model([model], [1], weights, play_frames, play_rounds, scene_file_name, reward_net=reward_net, use_expert=use_expert, return_path=return_path)


def play_multi_model(model_list, lamda_list, weights, play_frames=10000, play_rounds=10000, scene_file_name='scenes/scene-city.txt', reward_net=None, use_expert=False, return_path=False, max_step_1round=3000):

    # obs = [692.96580, 765.50164, 844.19555, 962.30012, 1111.86355, 1314.24535, 1602.83350, 1964.12510, 2422.89467, 2861.01471, 3290.37220, 3783.01214, 4081.32759, 4225.85403, 3327.85811, 2706.77085, 2181.52107, 1972.70783, 1841.97445, 1620.98429, 1357.73803, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 100.00000, 99.48760, 97.31135, 485.91348, 0.19696, 0.00000, 0.48562]
    # act = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # paths = [{'observations':obs, 'actions': act}, {'observations':obs, 'actions': act}]


    paths = []
    obs = []
    acts = []
    prob = []

    # init
    car_move = 0
    game_state = carmunk.GameState(weights, scene_file_name = scene_file_name, reward_net=reward_net, use_expert=use_expert)
    _, state, _, _, _ = game_state.frame_step((11))
    featureExp = np.zeros(NUM_FEATURES)
    round_num = 0
    score_list = []
    dist_list = []
    dist_1round = 0
    step_1round = 0
    step_1round_list = []
    

    time_list = []

    # start to move
    while True:
        start_time = timeit.default_timer()
        car_move += 1
        step_1round += 1


        action_onehot_encoded = np.zeros(NUM_ACTIONS)
        if use_expert:
            action = game_state.get_expert_action()
        else:
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
            action_onehot_encoded[action] = 1
            acts.append(action_onehot_encoded)


        #TODO
        #action = random.randrange(0, len(qval.flatten()))

        # take the action
        reward , next_state, readings, score, dist_1step = game_state.frame_step(action)
        dist_1round += dist_1step
        #print ("reward: ", reward)
        #print ("readings: ", readings)


        # acts.append(action)

        obs.append(readings)
        if not use_expert:
            prob.append(qval[0])
            # prob.append(qval)

        # start recording feature expectations only after 100 frames
        if car_move > 100:
            featureExp += (GAMMA**(car_move-101))*np.array(readings)
        #print ("featureExp: ", featureExp)

        # Tell us something.
        if readings[-1]==1 or step_1round==max_step_1round:
            step_1round_list.append(step_1round)
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

            agent_infos = {'prob': np.array(prob)}
            paths.append({'observations': np.array(obs), 'actions': np.array(acts), 'agent_infos': agent_infos})
            obs = []
            acts = []
            prob = []


        if play_frames > 0 and car_move % play_frames == 0:
            #print("The car has moved %d frames" % car_move)
            if readings[-1] == 0:
                round_num += 1
                score_list.append(score)
                dist_list.append(dist_1round)
                agent_infos = {'prob': np.array(prob)}
                paths.append({'observations': np.array(obs), 'actions': np.array(acts), 'agent_infos': agent_infos})
            #print("Score in this round: ", score)
            #print("Aver Score in ", round_num, "rounds: ", np.average(score_list))
            #print("Dist in this round: ", dist_1round)
            #print("Aver dist in ", round_num, "rounds: ", np.average(dist_list))
            if step_1round > 0:
                step_1round_list.append(step_1round)
            break

        if play_rounds > 0 and round_num == play_rounds:
            #print("Score in this round: ", score)
            #print("Aver Score in ", round_num, "rounds: ", np.average(score_list))
            #print("Dist in this round: ", dist_1round)
            #print("Aver dist in ", round_num, "rounds: ", np.average(dist_list))
            score_list.append(score)
            dist_list.append(dist_1round)
            if step_1round > 0:
                step_1round_list.append(step_1round)
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

    if return_path:
        return paths

    return featureExp, np.average(score_list), np.average(dist_list), np.average(step_1round_list)

if __name__ == "__main__": 
    #BEHAVIOR = sys.argv[1]
    #ITERATION = sys.argv[2]
    #FRAME = sys.argv[3]
    
    BEHAVIOR = "city"
    ITERATION = 20000
    FRAME = 1
    score_list = []
    dist_list = []
    
    for ROUND in range(1):
        for FRAME in range(1,20):
            print('***************************************************************************************************')
            print('FRAME ', FRAME)
            # FRAME = 4
            modelType = BEHAVIOR
            #model_dir = 'results/models-'+ modelType +'/'
            model_dir = 'results/finals/'
            model_dir = 'results/models-city/'
            model_dir = 'results/models-city_RL_reward0/'
            model_dir = 'results/models-city_RL_2layers_good/'
            # model_dir = 'results/models-city_RL_2layers_5actions_good/'
            # model_dir = 'results/models-city_AIRL_2layers0820_5actions/'
            # model_dir = 'results/models-city_AIRL_2layers/'
            # model_dir = 'results/models-city_GAIL_2layers/'
            # saved_model = model_dir+'164-150-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'.h5'
            saved_model = model_dir+'64-128-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'.h5'
            # saved_model = model_dir+'64-128-100-50000-'+str(ITERATION)+'-'+str(FRAME)+'_frame12000.h5'
            # saved_model = model_dir + '164-150-100-50000-200000-1_frame200000.h5'
            saved_model = model_dir + '164-150-100-50000-100000-1_frame'+str(FRAME*10000)+'.h5'
            saved_model = model_dir + '164-150-100-50000-200000-1_frame51000.h5'

            # saved_model = 'results/models-city_RL_2layers_good/164-150-100-50000-200000-1_frame51000.h5'
            saved_model = 'results/models-city_RL_2layers_good/164-150-100-50000-200000-1_frame112000.h5'

            saved_model = 'results/models-city_RL_2layers0823/164-150-100-50000-200000-1_frame'+str(FRAME*10000)+'.h5'


            saved_model = 'results/models-city_AIRL_2layers0819/164-150-100-50000-100000-0_frame14000.h5' # for AIRL

            # saved_model = 'results/models-city_GAIL_2layers/164-150-100-50000-20000-'+str(ROUND)+'_frame'+str(FRAME*1000)+'.h5'
            saved_model = 'results/models-city_GAIL_2layers/164-150-100-50000-20000-0_frame7000.h5' # for GAIL


            saved_model = 'results/models-city_RL_2layers0826_withexp/164-150-100-50000-200000-1_frame200000.h5' # for RL
            saved_model = 'results/models-city_RL_2layers0826_withexp/164-150-100-50000-200000-1_frame'+str(FRAME*1000)+'.h5' # for RL

            saved_model = 'results/models-city/164-150-100-50000-20000-'+str(FRAME)+'.h5' # for RL
            # if not os.path.exists(saved_model):
            #     continue


            weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
            model = net1(NUM_FEATURES, NUM_ACTIONS, [164, 150], saved_model)
            # model = net1(NUM_FEATURES, NUM_ACTIONS, [64, 128, 64, 32, 16, 8, 16, 32, 64, 128, 64], saved_model)
            
            # scene_file_name = 'scenes/scene-city-car.txt'
            # scene_file_name = 'scenes/scene-ground-car.txt'
            scene_file_name = 'scenes/scene-city.txt'
            reward_weights=''
            # featureExp, score, dist = play(model, weights=reward_weights, play_rounds=100, scene_file_name = scene_file_name)
            featureExp, score, dist = play(model, weights=reward_weights, play_rounds=3, play_frames=10000, scene_file_name = scene_file_name, use_expert=True)
            score_list.append(score)
            dist_list.append(dist)

            for feature in featureExp:
                print('{:.3f}'.format(feature), end =", ")

        print('***************************************************************************************************')
        print('Round ID ', ROUND)
        for i in range(len(score_list)):
            print(i+1, 'score', score_list[i], 'dist', dist_list[i])


