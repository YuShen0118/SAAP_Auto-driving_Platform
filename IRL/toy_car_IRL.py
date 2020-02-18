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
        norm = np.linalg.norm(weights, ord=1)
        weights = weights/norm
        print("Computing optimal weights finished!")
        return weights # return the normalized weights


    def UpdatePolicyFEList(self, weights, opt_count, scene_file_name):  
        # store feature expecations of a newly learned policy and its difference to the expert policy	
        print("Updating Policy FE list starts......")
        
        #start_time = timeit.default_timer()
        model_name = QLearning(num_features, num_actions, self.params, weights, self.results_folder, self.behavior_type, self.train_frames, opt_count, scene_file_name)	
        
        #print("Total consumed time: ", timeit.default_timer() - start_time, " s")
            
        # get the trained model
        print("The latest Q-learning model is: ", model_name)
        model = net1(self.num_features, self.num_actions, self.params['nn'], model_name)
        
        # get feature expectations by executing the learned model
        temp_fe, aver_score = play(model, weights, self.play_frames, play_rounds=10)
        
        # t = (weights.tanspose)*(expertFE-newPolicyFE)
        # hyperdistance = t
        temp_hyper_dis = np.abs(np.dot(weights, np.asarray(self.expert_fe)-np.asarray(temp_fe))) 
        self.policy_fe_list[temp_hyper_dis] = temp_fe
        
        print("Updating Policy FE list finished!")
        return temp_hyper_dis, aver_score
        
        
    def IRL(self, scene_file_name):
        # create a folder for storing results
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        
        # create a file to store weights after each iteration of learning
        f = open(self.results_folder + 'weights-'+self.behavior_type+'.txt', 'w')
        nearest_dist = 9999999999
        nearest_iter_no = -1
        opt_count = 1
        while True:
            print("================ IRL iteration number: ", opt_count, " ================")
            
            # Main Step 1: compute the new weights according to the list of policies and the expert policy
            weights_new = self.ComputeOptimalWeights() 
            print("The optimal weights so far: ", weights_new)
            f.write( str(weights_new) )
            f.write('\n')
            
            # Main Step 2: update the policy feature expectations list
            # and compute the distance between the lastest policy and expert feature expecations
            self.current_dis, score = self.UpdatePolicyFEList(weights_new, opt_count, scene_file_name)
            f1 = open(self.results_folder + 'models-'+ behavior_type +'/' + 'results.txt', 'a')
            f1.write("iteration " + str(opt_count) + ": current_dis " +str(self.current_dis) + "  score " + str(score))
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
    
    #random_fe = [45.16965, 51.62903, 55.84101, 64.65182, 74.78856, 90.46633, 113.45914, 137.13300, 152.57598, 162.98980, 180.00261, 198.95925, 228.79739, 267.95084, 327.91051, 428.06988, 451.92122, 263.78044, 152.64403, 130.79640, 116.01192, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 4.97997, 4.95111, 4.80826, 5.00000, 5.00000, 5.00000, 42.62056, 0.05593, 15.63755, 0.01078]
    #expert_city_fe = [52.72018, 57.54477, 62.24114, 69.71412, 76.95464, 90.42736, 106.56914, 129.54432, 157.43855, 206.61816, 239.03757, 256.96736, 284.76613, 323.39457, 378.64623, 410.54772, 433.42094, 194.71432, 130.74410, 113.02894, 99.19676, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 4.99958, 4.99978, 4.99962, 4.99965, 4.99957, 4.98407, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 39.04116, 0.00493, 16.95020, 0.00000]
    random_fe = [81.51861, 92.91939, 108.32393, 129.20138, 155.07127, 185.09527, 228.15942, 297.12296, 345.95401, 392.92691, 458.90575, 504.89901, 551.83029, 383.31081, 174.93949, 128.19650, 107.53420, 92.20552, 85.07736, 79.42055, 76.33213, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 30.75084, 0.00000, 10.39512, 0.00000, 0.00000]
    expert_city_fe = [170.65782, 190.90100, 218.28805, 253.84296, 302.35227, 334.81273, 374.84139, 424.95082, 462.06879, 501.20787, 196.24086, 147.47648, 111.49499, 89.36053, 72.28600, 56.71287, 45.34124, 36.20385, 34.40675, 32.21060, 31.20924, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 5.00000, 9.94075, 0.00000, 7.43905, 0.00000, 0.00000]

    # training parameters
    #nn_param = [164, 150]
    nn_param = [64, 128, 256, 128, 64]
    params = {
        "batch_size": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    epsilon = 0.1 # termination when t<0.1
    num_features = 47
    num_actions = 25
    train_frames = 20000   # number of RL training frames per iteration of IRL
    play_frames = 2000 # the number of frames we play for getting the feature expectations of a policy online
    behavior_type = 'city' # yellow/brown/red/bumping
    results_folder = 'results/'

    scene_file_name = 'scenes/scene-city.txt'

    irl_learner = IRLAgent(params, random_fe, expert_city_fe, epsilon, \
                            num_features, num_actions, train_frames, play_frames, \
                            behavior_type, results_folder)
    irl_learner.IRL(scene_file_name)

