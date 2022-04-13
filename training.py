#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Policy Gradient Algorithms for Risk-Sensitive Reinforcement Learning
    Christos Mavridis, Erfaun Noorani, John Baras <{mavridis,enoorani,baras}@umd.edu>
    Department of Electrical and Computer Engineering 
    University of Maryland
    March 2022
"""

#%% Importing Modules

"""
    Dependencies: PyTorch (Neural Network Training), Gym (RL environments)
"""

import PG

#%% Hyper-parameters 

# Name of the file to be saved
name = '' # all parameters are shown in the name
# name = 'test' # overwrite name with string

# Game
game='cartpole'
# game='acrobot'
model_var = [round(0.2*(i+1),1) for i in range(10)] # variations of the model to be tested

# Risk Sensitivity
risk_beta = -0.5
risk_objective = 'BETA' # 'BETA', 'BETAI', 'BETAS' 

# Simulation Epochs
train_loops = 100 # hundreds of training epochs
test_loops = 50 # hundreds of testing epochs
nepochs = 10
time_steps = 200 # time horizon for successful epoch
gammaAC = 0.99

# Training Model 
nn_hidden_size = [16] # NN layers
lr = 0.01 # NN learning rate
cut_lr=False # Reduce lr if goal is achieved 

# REINFORCE / Look-Ahead / TD
look_ahead = 100 # 0 for REINFORCE, 1 for TD learning, number for look_ahead
baseline = False # remove baseline (True) or not; TD learning always uses baseline

# Stepsizes
a_outer = 0.0 # stochastic approximation stepsizes updated every episode; 0 means constant
a_inner = 0.5 # stochastic approximation stepsizes updated every observation; 0 means constant

# random seed
rs=0

#%%

for lr in [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05]:
    for a_inner in [0.1,0.5]:
        
        PG.train(name=name, 
                game=game,
                model_var = model_var,
                risk_beta = risk_beta,
                risk_objective = risk_objective, 
                train_loops = train_loops,
                test_loops = test_loops,
                nepochs = nepochs,
                time_steps = time_steps, 
                gammaAC = gammaAC,
                nn_hidden_size = nn_hidden_size,
                lr = lr,
                cut_lr=cut_lr,
                look_ahead = look_ahead,
                baseline = baseline,
                a_outer = a_outer,
                a_inner = a_inner, 
                rs=rs)
