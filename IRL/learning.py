from simulation import carmunk
import numpy as np
import random
import csv
from neuralNets import net1, LossHistory
import os
import os.path
import timeit
from playing import play
from keras import backend as K
import keras


GAMMA = 0.99  # discount factor
TUNING = False  # If False, just use arbitrary, pre-selected params


def params_to_filename(params): 
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batch_size']) + '-' + str(params['buffer'])
            
'''            
def IRLHelper(weights, behavior_type, train_frames, opt_count):
    nn_param = [164, 150]
    params = {
        "batch_size": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    model = net1(NUM_FEATURES, NUM_ACTIONS, nn_param)
    train_net(model, params, weights, behavior_type, train_frames, opt_count)
'''            

def outPutW(weights, border=4):
    for w in weights:
        if len(w.shape) == 1:
            print(w[1:border])
        if len(w.shape) == 2:
            print(w[1:border, 1:border])
        if len(w.shape) == 3:
            print(w[1:border, 1:border, 1:border])
            
            
def QLearning(num_features, num_actions, params, weights, results_folder, behavior_type, train_frames, opt_count, scene_file_name, 
              continue_train=False, hitting_reaction_mode=2, enlarge_lr=0, reward_net=None):
    '''
    The goal of this function is to train a function approximator of Q which can take 
    a state (eight inputs) and predict the Q values of three actions (three outputs)
    '''
    print("Q learning starts...")
    
    # init variables
    epsilon = 1 # the threshold for choosing a random action over the best action according to a Q value
    if continue_train:
        epsilon = 0.5
    d_epsilon = epsilon / train_frames
    observe_frames = 100  # we train our first model after observing certain frames
    replay = []  # store tuples of (state, action, reward, next_state) for training 
    survive_data = [] # store how long the car survived until die
    loss_log = [] # store the train loss of each model
    score_log = [] # store the train loss of each model
    dist_log = [] # store the train loss of each model
    my_batch_size = params['batch_size']
    buffer = params['buffer'] 
    assert (observe_frames >= my_batch_size), "Error: The number of observed frames is less than the batch size!"
    
    # create a folder and process the file name for saving trained models
    model_dir = results_folder + 'models-'+ behavior_type +'/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = params_to_filename(params) + '-' + str(train_frames) + '-' + str(opt_count)
    model_name = model_dir + filename + '.h5' 
    weights_name = model_dir + filename + '_weights.npy'

    pretrained_model = ''
    if continue_train and (opt_count > 1):
        pretrained_model = model_dir + params_to_filename(params) + '-' + str(train_frames) + '-' + str(opt_count-1) + '.h5' 

    # init a neural network as an approximator for Q function
    epochCount = 1
    if continue_train:
        epochCount = opt_count
    model = net1(num_features, num_actions, params['nn'], weightsFile=pretrained_model, epochCount=epochCount, enlarge_lr=enlarge_lr)
     
    # create a new game instance and get the initial state by moving forward
    game_state = carmunk.GameState(weights, scene_file_name, reward_net=reward_net, action_num=num_actions)
    _, state, _, _, _ = game_state.frame_step((11))
    #_, state, _ = game_state.frame_step((0,1))

    # let's time it
    start_time = timeit.default_timer()

    expert_count = 0

    stop_status = 0

    # run the frames
    frame_idx = 0
    car_move_count = 0     # track the number of moves the car is making
    car_surivive_move_count = 0 # store the maximum moves the car made before run into something
    print("In QLearning - the total number of training frames is: ", train_frames)
    while frame_idx < train_frames:
        
        if frame_idx % 1000 == 0:
            print("In QLearning - current training frame is: ", frame_idx)
        
        frame_idx += 1
        car_move_count += 1

        # choose an action.
        # before we reach the number of observing frame (for training) we just sample random actions
        if expert_count > 0:
            action = game_state.get_expert_action()
            expert_count -= 1
        elif random.random() < epsilon or frame_idx < observe_frames:
            action = np.random.randint(0, num_actions)  # produce action 0, 1, or 2
            #action = np.random.random([2])*2-1
        else:
            # get Q values for each action. Q values are scores associated with each action (there are in total 3 actions)
            qval = model.predict(state, batch_size=1)
            action = (np.argmax(qval))  # get the best action
            #action = model.predict(state, batch_size=1)

        # execute action, receive a reward and get the next state
        reward, next_state, _, _, _ = game_state.frame_step(action, hitting_reaction_mode = hitting_reaction_mode)
        if hitting_reaction_mode == 2: # use expert when hitting
            if next_state[0][-1] == 1: # hitting
                if expert_count == 0:
                    expert_count = game_state.max_history_num
                else:
                    expert_count = 0

        # store experiences
        replay.append((state, action, reward, next_state))

        # if we're done observing, start training
        if frame_idx > observe_frames:

            # If we've stored enough in our buffer, pop the oldest 
            if len(replay) > buffer: # currently buffer = 50000
                replay.pop(0)

            # sample our experience
            mini_batch = random.sample(replay, my_batch_size) # currently batchSize = 100

            # get training data
            X_train, y_train = process_minibatch(mini_batch, model, num_features, num_actions)

            # train a model on this batch
            history = LossHistory()
            model.fit(X_train, y_train, batch_size=my_batch_size, epochs=1, verbose=0, callbacks=[history])

            #outPutW(model.get_weights())

            loss_log.append(history.losses)
            if frame_idx % 100 == 0:
                print("history.losses ", history.losses)
                
            if frame_idx % 1000 == 0:
                temp_fe, aver_score, aver_dist = play(model, weights, play_rounds=10, scene_file_name=scene_file_name)
                if len(score_log) == 0 or (len(score_log) > 0 and aver_score > np.max(score_log) and aver_dist > np.max(dist_log)):
                    model.save_weights(model_name.replace('.h5', '_frame'+str(frame_idx)+'.h5'), overwrite=True)
                    np.save(weights_name, weights)
                    print("Saving model inner: ", model_name)
                score_log.append([aver_score])
                dist_log.append([aver_dist])
                
            '''
            if frame_idx % 4000 == 0:
                lr = 0.001 / 2**(frame_idx/4000)
                print('===============lr===============', lr)

                #optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
                #optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
                optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
                #optimizer = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
                #optimizer = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
                model.compile(optimizer=optimizer, loss='mse')
            '''

            # diverges, early stop
            '''
            if history.losses[0] > 1000:
                model = net1(num_features, num_actions, params['nn'], weightsFile=pretrained_model)
                model.save_weights(model_name, overwrite=True)
                np.save(weights_name, weights)
                print("Diverges, early stop, loss=", history.losses[0])
                print("Saving model: ", model_name)
                stop_status = -1
                break

            #converges, early stop
            if history.losses[0] < 1e-6:
                model.save_weights(model_name, overwrite=True)
                np.save(weights_name, weights)
                print("Converges, early stop, loss=", history.losses[0])
                print("Saving model: ", model_name)
                stop_status = 1
                break
            '''

        # update the state
        state = next_state

        # decrease epsilon over time to reduce the chance taking a random action over the best action based on Q values
        if epsilon > 0.1 and frame_idx > observe_frames:
            epsilon -= d_epsilon

        # car died, update
        if state[0][-1] == 1:
            # log the car's distance at this frame index 
            survive_data.append([frame_idx, car_move_count])

            # update
            if car_move_count > car_surivive_move_count:
                car_surivive_move_count = car_move_count

            # time it
            survive_time = timeit.default_timer() - start_time
            fps = car_move_count / survive_time

            # reset
            car_move_count = 0
            start_time = timeit.default_timer()

        # save the current model 
        if frame_idx == train_frames:
            model.save_weights(model_name, overwrite=True)
            np.save(weights_name, weights)
            print("Saving model: ", model_name)

    # log results after we're done with all training frames
    log_results(results_folder, filename, survive_data, loss_log, score_log, dist_log)
    print("Q learning finished!")

    # K.clear_session()
    return model_name, stop_status
    
    

def log_results(results_folder, filename, survive_data, loss_log, score_log, dist_log):
    log_dir = results_folder + 'models-city_RL_2layers0907_withexp/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # save the results to a file so that we can graph it later
    with open(log_dir + 'survive_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump, lineterminator = '\n')
        wr.writerows(survive_data)

    with open(log_dir + 'training_loss-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf, lineterminator = '\n')
        for loss_item in loss_log:
            wr.writerow(loss_item)
            
    with open(log_dir + 'training_score-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf, lineterminator = '\n')
        for item in score_log:
            wr.writerow(item)
            
    with open(log_dir + 'training_trajectory_length-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf, lineterminator = '\n')
        for item in dist_log:
            wr.writerow(item)


def process_minibatch(minibatch, model, num_features, num_actions):  
    X_train = [] # states of the agent
    y_train = [] # three actions and their corresponding rewards
    
    # create a training dataset
    for memory in minibatch:
        # get stored values
        state_m, action_m, reward_m, next_state_m = memory
        
        # get prediction on the current state.
        currentQ = model.predict(state_m, batch_size=1)
        
        # get prediction on the next state.
        nextQ = model.predict(next_state_m, batch_size=1)
        
        # get our best move
        maxQ = np.max(nextQ)
        y = np.zeros((1, num_actions)) 
        y[:] = currentQ[:]
        
        # check for terminal state
        if next_state_m[0][-1] == 1:  # the terminal state
            update = reward_m
        else:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        
        # update the Q value for the action we take on the current state (i.e., state_m)
        y[0][action_m] = update
        X_train.append(state_m.reshape(num_features,))
        y_train.append(y.reshape(num_actions,)) 

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

'''
def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)

    result_dir = RESULTS_DIR + 'logs/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    if not os.path.isfile(result_dir + 'loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open(result_dir + 'loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = net1(NUM_FEATURES, NUM_ACTIONS, params['nn'])
        train_net(model, params)
    else:
        print("Already tested.")
'''

if __name__ == "__main__":
    print("In learning.py")
    '''
    weights = [ 0.04924175 ,-0.36950358 ,-0.15510825 ,-0.65179867 , 0.2985827 , -0.23237454 , 0.21222881 ,-0.47323531]
    model_type = 'default'
    if TUNING:
        param_list = []
        nn_param_set = [[164, 150], [256, 256], [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_param_set:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batch_size": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)
    else:
        nn_param = [164, 150]
        params = {
            "batch_size": 100,
            "buffer": 50000,
            "nn": nn_param
        }
        model = net1(NUM_FEATURES, NUM_ACTIONS, nn_param)
        train_frames = 1000
        train_net(model, params, weights, model_type, train_frames)
    '''
