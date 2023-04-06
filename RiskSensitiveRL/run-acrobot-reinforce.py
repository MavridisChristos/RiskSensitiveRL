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

#%% Hyper parameters

risk_betas = [0,0.01,0.05,0.1,-0.01,-0.05,-0.1]
learning_rates = [0.001,0.003,0.005,0.007,0.01]
b = 0
random_seeds = [b+0,b+1,b+2,b+3,b+4,b+5,b+6,b+7,b+8,b+9]
 
#%% 

def run_all_betas(
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
                rs):
    
    for risk_beta in risk_betas:
        
        train(
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

#%% Main

if __name__ == "__main__":

    # Name and Game
    name = '' 
    game='acrobot'
    # REINFORCE or Actor-Critic 
    look_ahead = 0
    baseline = False
    # Risk-Sensitivity
    risk_beta = 0
    risk_objective = 'BETA' 
    gammaAC = 0.99
    # Training Loops
    train_loops = 200
    test_loops = 100
    nepochs = 10
    time_steps = 200 
    # Neural Networks and Learning Rate
    nn_actor = [64]
    nn_critic = [64]
    lr = 0.0007
    a_outer = 0.0
    a_inner = 0.0 
    cut_lr=1
    # Model Variations
    model_var = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    verbose = 0
    rs=0
    
    p=[]
    for lr in learning_rates:
        for rs in random_seeds:
            p.append(multiprocessing.Process(target=run_all_betas, args=(
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
            print(f'Process #{len(p)} started.')
    
    # for pp in p:
    #     pp.join()
    print('Parent Process Terminated.')
    
#%%

"""
    Policy Gradient Algorithms for Risk-Sensitive Reinforcement Learning
    Christos Mavridis, Erfaun Noorani, John Baras <{mavridis,enoorani,baras}@umd.edu>
    Department of Electrical and Computer Engineering 
    University of Maryland
"""
