'''
IRL algorithm developed for the toy car obstacle avoidance problem
'''

import numpy as np
import logging
import scipy
from playing import play            # get the RL Test agent, gives out feature expectations after 2000 frames
from neuralNets import net1         # construct the nn and send to playing
from cvxopt import matrix, solvers  # convex optimization library
from learning import QLearning      # get the Reinforcement learner
import os
import timeit


start_time = timeit.default_timer()

class IRLAgent:
    def __init__(self, params, random_fe, expert_fe, epsilon, num_features, num_actions, train_frames, play_frames, behavior_type, results_folder):
        self.params = params
        self.random_fe = random_fe
        self.expert_fe = expert_fe
        self.epsilon = epsilon 
        self.num_features = num_features
        self.num_actions = num_actions
        self.train_frames = train_frames
        self.play_frames = play_frames 
        self.behavior_type = behavior_type
        self.results_folder = results_folder
        self.random_dis = np.linalg.norm(np.asarray(self.expert_fe)-np.asarray(self.random_fe)) # norm of the diff between the expert policy and random policy
        self.policy_fe_list = {self.random_dis:self.random_fe} # storing the policies and their respective t values in a dictionary
        print ("Expert - Random's distance at the beginning: " , self.random_dis) 
        self.current_dis = self.random_dis
    
    
    def ComputeOptimalWeights(self): 
        # implement the convex optimization, posed as an SVM problem
        print("Computing optimal weights starts......")
        m = len(self.expert_fe)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')

        # add the feature expectations of the expert policy
        policy_fe_list = [self.expert_fe]
        h_list = [1]

        # add the feature expectations of other policies
        for i in self.policy_fe_list.keys():
            policy_fe_list.append(self.policy_fe_list[i])
            h_list.append(1)
            
        # The resulting weights dot product expert police feature's expectations should be >= 1
        # The resulting weights dot product other policies' feature expectations should be <= -1
        policy_fe_mat = np.matrix(policy_fe_list)
        policy_fe_mat[0] = -1*policy_fe_mat[0] # reverse the expert policy computation

        G = matrix(policy_fe_mat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        print("Computing optimal weights finished!")
        return weights # return the normalized weights

    
    def UpdatePolicyFEList(self, weights, opt_count, scene_file_name, enlarge_lr):  
        # store feature expecations of a newly learned policy and its difference to the expert policy	
        print("Updating Policy FE list starts......")
        
        #start_time = timeit.default_timer()
        model_name, stop_status = QLearning(num_features, num_actions, self.params, weights, self.results_folder, self.behavior_type, self.train_frames, opt_count, scene_file_name, enlarge_lr=enlarge_lr)	
        
        #print("Total consumed time: ", timeit.default_timer() - start_time, " s")
            
        # get the trained model
        print("The latest Q-learning model is: ", model_name)
        model = net1(self.num_features, self.num_actions, self.params['nn'], model_name)
        
        # get feature expectations by executing the learned model
        temp_fe, aver_score, aver_dist = play(model, weights, self.play_frames, play_rounds=10, scene_file_name=scene_file_name)
        
        # t = (weights.tanspose)*(expertFE-newPolicyFE)
        # hyperdistance = t
        temp_hyper_dis = np.abs(np.dot(weights, np.asarray(self.expert_fe)-np.asarray(temp_fe))) 
        self.policy_fe_list[temp_hyper_dis] = temp_fe
        
        print("Updating Policy FE list finished!")
        return temp_hyper_dis, aver_score, aver_dist, stop_status
        
        
        
    def IRL(self, scene_file_name):
        # create a folder for storing results
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        
        # create a file to store weights after each iteration of learning
        f = open(self.results_folder + 'weights-'+self.behavior_type+'.txt', 'w')
        nearest_dist = 9999999999
        nearest_iter_no = -1
        opt_count = 1
        enlarge_lr = 0
        while True:
            print("================ IRL iteration number: ", opt_count, " ================")
            
            # Main Step 1: compute the new weights according to the list of policies and the expert policy
            weights_new = self.ComputeOptimalWeights() 
            print("The optimal weights so far: ", weights_new)
            f.write( str(weights_new) )
            f.write('\n')
            
            # Main Step 2: update the policy feature expectations list
            # and compute the distance between the lastest policy and expert feature expecations
            self.current_dis, score, car_dist, stop_status = self.UpdatePolicyFEList(weights_new, opt_count, scene_file_name, enlarge_lr)
            if stop_status == 1:
                enlarge_lr += 1
            f1 = open(self.results_folder + 'models-'+ behavior_type +'/' + 'results.txt', 'a')
            f1.write("iteration " + str(opt_count) + ": current_dis " +str(self.current_dis) + "  score " + str(score) + "  trajectory length " + str(car_dist))
            f1.write('\n')
            f1.close()
            
            # Main Step 3: assess the above-computed distance, decide whether to terminate IRL
            print("The stopping distance thresould is: ", epsilon)
            print("The latest policy to expert policy distance is: ", self.current_dis)
            
            if nearest_dist > self.current_dis:
                nearest_dist = self.current_dis
                nearest_iter_no = opt_count
            print("So far the nearest dist is: ", nearest_dist, ", in the", nearest_iter_no, "th iteration")

            print("Total consumed time: ", timeit.default_timer() - start_time, " s")
            print("Total consumed time: ", (timeit.default_timer() - start_time)/3600.0, " h")
            print("===========================================================")
            print("\n")
            if self.current_dis <= self.epsilon: 
                print("IRL finished!")
                print("The final weights of IRL is: ", weights_new)
                break
            opt_count += 1
        f.close()



                
            
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # policy feature expectations
    random_fe = [7.74363107, 4.83296402, 6.1289194, 0.39292849, 2.0488831, 0.65611318, 6.90207523, 2.46475348]
    expert_yellow_fe = [7.5366e+00, 4.6350e+00, 7.4421e+00, 3.1817e-01, 8.3398e+00, 1.3710e-08, 1.3419e+00, 0.0000e+00]
    expert_red_fe = [7.9100e+00, 5.3745e-01, 5.2363e+00, 2.8652e+00, 3.3120e+00, 3.6478e-06, 3.82276074e+00, 1.0219e-17] 
    expert_brown_fe = [5.2210e+00,  5.6980e+00,  7.7984e+00, 4.8440e-01, 2.0885e-04, 9.2215e+00, 2.9386e-01, 4.8498e-17]
    expert_bumping_fe = [7.5313e+00, 8.2716e+00, 8.0021e+00, 2.5849e-03, 2.4300e+01, 9.5962e+01, 1.5814e+01, 1.5538e+03]
    expert_me1_fe = [7.61899296e+00, 5.57997070e+00, 4.05467547e+00, 7.06288984e-01, 8.47292102e-01, 2.08010868e-12, 8.44641891e+00, 0.00000000e+00]
    
    random_fe = [45.16965, 51.62903, 55.84101, 64.65182, 74.78856, 90.46633, 113.45914, 137.13300, 152.57598, 162.98980, 180.00261, 198.95925, 228.79739, 267.95084, 327.91051, 428.06988, 451.92122, 263.78044, 152.64403, 130.79640, 116.01192, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 4.97997, 4.95111, 4.80826, 5.00000, 5.00000, 5.00000, 42.62056, 0.05593, 15.63755, 0.01078]
    expert_city_fe = [52.72018, 57.54477, 62.24114, 69.71412, 76.95464, 90.42736, 106.56914, 129.54432, 157.43855, 206.61816, 239.03757, 256.96736, 284.76613, 323.39457, 378.64623, 410.54772, 433.42094, 194.71432, 130.74410, 113.02894, 99.19676, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 4.99958, 4.99978, 4.99962, 4.99965, 4.99957, 4.98407, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 39.04116, 0.00493, 16.95020, 0.00000]

    expert_scene1_no_norm_fe = [338.33755, 344.27074, 680.85631, 658.40258, 466.42694, 645.92039, 673.47171, 692.37062, 718.96518, 577.21300, 808.27076, 878.30508, 970.64749, 999.48579, 999.99309, 999.96561, 523.95592, 319.02887, 263.13024, 225.22845, 776.58537, 7.19367, 7.35936, 9.98863, 9.84560, 6.73880, 9.66667, 10.00000, 10.00000, 9.99933, 6.66667, 10.00000, 10.00000, 10.00000, 0.19476, 0.00210, 0.00827, 4.46883, 6.40949, 6.33195, 6.52312, 1.85568, 0.00000, 0.00000, 0.03755, 0.00000]
    random_scene1_no_norm_fe = [829.29850, 878.28727, 843.86715, 768.74765, 753.03655, 719.61687, 696.61013, 649.52254, 594.09799, 656.43592, 438.23509, 499.64753, 726.45449, 768.82468, 820.95955, 892.30909, 976.95597, 851.23313, 987.10134, 962.82460, 664.07779, 9.99697, 9.78596, 9.14821, 8.63431, 9.99879, 9.99649, 9.99595, 9.30154, 8.33287, 9.79282, 7.67933, 8.07953, 9.94465, 10.00000, 9.99990, 9.99962, 7.03683, 4.75631, 0.46347, 0.62436, 5.22953, 57.20994, 0.01675, 0.03718, 0.00005]

    expert_scene2_no_norm_fe = [129.22780, 136.78742, 147.62074, 157.36085, 175.28098, 196.63826, 220.52759, 248.90806, 279.48842, 313.95628, 350.18217, 383.07786, 421.35470, 430.55552, 307.78473, 200.84458, 70.27693, 51.98201, 42.33212, 34.82107, 31.05633, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 45.56306, 0.00000, 0.03011, 0.00000]
    random_scene2_no_norm_fe = [51.83432, 60.30839, 68.53151, 80.81281, 101.47611, 128.34691, 162.23168, 210.05810, 273.98098, 343.02684, 403.94875, 465.96357, 525.60607, 573.44138, 413.20720, 194.78817, 150.16052, 129.66170, 113.42596, 103.01873, 98.70103, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 9.99998, 40.49556, 0.00000, 0.03601, 0.00000]

    expert_scene3_no_norm_fe = [142.57352, 166.78612, 200.42974, 248.83981, 289.54972, 325.12905, 373.26140, 432.61656, 468.52714, 471.85909, 557.23623, 153.11445, 117.48357, 98.50721, 82.95734, 74.75753, 68.43096, 63.54403, 60.97443, 58.94622, 58.96246, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 9.93355, 5.48316, 5.01861, 5.03865, 5.01324, 5.00027, 5.00004, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 0.25784, 0.00000, 0.03752, 0.00000]
    random_scene3_no_norm_fe = [78.53235, 88.95513, 101.45447, 120.38489, 143.80693, 172.55931, 211.16029, 269.38707, 328.87681, 371.84652, 431.91123, 487.87203, 524.27638, 509.30660, 208.32369, 146.05923, 115.66023, 99.24772, 87.44452, 82.38363, 76.64021, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 5.00001, 6.01284, 9.43338, 5.20623, 5.00039, 5.00033, 5.00015, 5.00009, 5.00005, 5.00004, 34.93628, 0.00000, 0.04136, 0.00001]

    expert_scene1_norm_fe = [7.17438, 6.46819, 4.58467, 3.39507, 5.33102, 6.63920, 6.73472, 6.92371, 7.18995, 7.55921, 8.08271, 7.61623, 9.70637, 9.51901, 9.99984, 10.00000, 3.35378, 5.82030, 9.57222, 9.92083, 9.97734, 9.85797, 9.53387, 8.05100, 7.18372, 8.87797, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 7.47432, 9.99981, 6.69433, 0.00194, 0.00002, 6.16775, 3.92805, 0.37554, 0.06668, 0.01865, 0.00000, 0.00000, 0.03755, 0.00000]
    random_scene1_norm_fe = [6.01234, 5.67800, 6.35946, 6.65966, 7.05836, 7.60069, 8.29843, 8.98605, 8.92283, 8.38116, 7.96379, 7.66569, 7.47672, 7.37341, 7.22399, 6.05623, 7.59192, 3.13917, 2.46684, 6.31021, 9.28537, 9.82506, 7.35277, 9.99793, 9.99800, 9.98701, 9.98817, 9.95851, 9.68698, 10.00000, 10.00000, 9.99884, 9.99366, 9.98283, 9.96277, 9.69001, 7.38002, 10.00000, 7.08953, 6.66767, 8.70832, 9.90721, 10.00000, 0.00000, -0.00198, 0.00000]

    expert_scene2_norm_fe = [1.29228, 1.36787, 1.47621, 1.57361, 1.75281, 1.96638, 2.20528, 2.48908, 2.79488, 3.13956, 3.50182, 3.83078, 4.21355, 4.30556, 3.07785, 2.00845, 0.70277, 0.51982, 0.42332, 0.34821, 0.31056, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 4.55631, 0.00000, 0.03011, 0.00000]
    random_scene2_norm_fe = [0.29995, 0.31014, 0.34212, 0.41839, 0.48016, 0.58340, 0.72743, 0.91747, 1.25017, 1.85084, 2.71814, 3.79510, 4.78585, 5.51194, 6.22807, 5.87311, 2.87438, 1.84570, 1.62929, 1.44398, 1.34130, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 5.88659, 0.00000, 0.01243, 0.00000]

    expert_scene3_norm_fe = [1.42574, 1.66786, 2.00430, 2.48840, 2.89550, 3.25129, 3.73261, 4.32617, 4.68527, 4.71859, 5.57236, 1.53114, 1.17484, 0.98507, 0.82957, 0.74758, 0.68431, 0.63544, 0.60974, 0.58946, 0.58962, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 9.93355, 5.48316, 5.01861, 5.03865, 5.01324, 5.00027, 5.00004, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 0.02578, 0.00000, 0.03752, 0.00000]
    random_scene3_norm_fe = [0.36357, 0.39925, 0.45543, 0.52319, 0.61193, 0.74234, 0.93921, 1.23736, 1.83733, 2.67268, 3.66366, 4.57498, 5.27648, 4.47506, 2.19105, 1.09935, 0.57707, 0.43996, 0.41455, 0.43089, 0.63150, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 6.35501, 8.58788, 9.64641, 9.85931, 9.92374, 9.96307, 9.95680, 8.61283, 5.10218, 0.00000, 0.01673, 0.00001]

    # training parameters
    nn_param = [164, 150]
    params = {
        "batch_size": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    epsilon = 0.1 # termination when t<0.1
    num_features = 46
    num_actions = 25
    train_frames = 20000   # number of RL training frames per iteration of IRL
    play_frames = 2000 # the number of frames we play for getting the feature expectations of a policy online
    behavior_type = 'city' # yellow/brown/red/bumping
    results_folder = 'results/'
    
    scene_file_name = 'scenes/scene-city-car.txt'
    scene_file_name = 'scenes/scene-ground-car.txt'
    scene_file_name = 'scenes/scene-city.txt'

    irl_learner = IRLAgent(params, random_scene2_norm_fe, expert_scene2_norm_fe, epsilon, \
                            num_features, num_actions, train_frames, play_frames, \
                            behavior_type, results_folder)
    irl_learner.IRL(scene_file_name)

