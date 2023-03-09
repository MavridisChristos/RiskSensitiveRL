#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Policy Gradient Algorithms (from REINFORCE to Actor-Critic)
        for Risk-Sensitive Reinforcement Learning
    
    Christos Mavridis, Erfaun Noorani, John Baras <{mavridis,enoorani,baras}@umd.edu>
    Department of Electrical and Computer Engineering 
    University of Maryland
"""

from rsrl import train
import multiprocessing
 
#%% Run as standalone program

if __name__ == "__main__":

    # training_all, testing_all, wa, wc = train()
    
    # Default Parameters
    
    # Name and Game
    name = '' 
    game='cartpole'
    # REINFORCE or Actor-Critic 
    look_ahead = 1
    baseline = False
    # Risk-Sensitivity
    risk_beta = 0
    risk_objective = 'BETAI' 
    gammaAC = 0.99
    # Training Loops
    train_loops = 150
    test_loops = 50
    nepochs = 10
    time_steps = 200 
    # Neural Networks and Learning Rate
    nn_actor = [16]
    nn_critic = [16]
    lr = 0.0007
    a_outer = 0.0
    a_inner = 0.0 
    cut_lr=False
    # Model Variations
    model_var = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    verbose = 0
    rs=43
    
    p=[]
    for risk_beta in [0,0.001,-0.001,0.005,-0.005,0.01,-0.01,0.05,-0.05]:
        for lr in [0.0001,0.0003,0.0005,0.0007,0.001,0.005,0.01]:
            for rs in [0,1,2,3,4,5,6,7,8,9]:
                p.append(multiprocessing.Process(target=train, args=(
                                                                    name, 
                                                                    game,
                                                                    look_ahead,
                                                                    baseline,
                                                                    risk_beta,
                                                                    risk_objective, 
                                                                    gammaAC,
                                                                    train_loops,
                                                                    test_loops,
                                                                    nepochs,
                                                                    time_steps, 
                                                                    nn_actor,
                                                                    nn_critic,
                                                                    lr,
                                                                    a_outer,
                                                                    a_inner, 
                                                                    cut_lr,
                                                                    model_var,
                                                                    verbose,
                                                                    rs
                                                                    )
                                                )
                        )
                p[-1].start()
        
    # for pp in p:
    #     pp.join()
    
#%%

"""
    Policy Gradient Algorithms for Risk-Sensitive Reinforcement Learning
    Christos Mavridis, Erfaun Noorani, John Baras <{mavridis,enoorani,baras}@umd.edu>
    Department of Electrical and Computer Engineering 
    University of Maryland
"""