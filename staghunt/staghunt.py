#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    Stag-Hunt Game model
    Erfaun Noorani, Mavridis Christos, John S. Baras 
    Department of Electrical and Computer Engineering 
    University of Maryland
    <enoorani,mavridis,baras>@umd.edu
    May 2022
"""

import numpy as np

#%% 

class staghunt():
    
    """
    Description:

    Source:

    State Variables:
        
    Hidden Variables (internal model variables that do not appear in the state):
        
    Control Variables:
        
    Reward:
        
    Initialization:
        See init function

    Episode Termination:

    """

    #%%
    def __init__(self):     
        
        self.actions = [0,1]
        state1 = 1
        state2 = 1
        self.state = np.array([state1,state2])
        
        self.state_size = len(self.state)
        self.action_size = len(self.actions)
        
        self.payoff = [[[5,5],[0,4]],[[4,0],[2,2]]]
        
        # Set seed
        self.seed()
     
    #%%    
    def step(self, action1, action2):

        self.state[0] = action1
        self.state[1] = action2
        
        error = bool(
            action1 < 0 
            or action2 < 0 
        )

        if not error:
            reward1,reward2 = self.reward(action1,action2)
        else:
            reward1 = -1
            reward2 = -1
        
        return self.state, reward1, reward2, error

    #%%
    def reward(self,action1,action2):
        
        reward1,reward2 = self.payoff[action1][action2]
        
        return reward1,reward2 

    #%%
    def reset(self):
        
        state1 = 1
        state2 = 1
        self.state = np.array([state1,state2])
        
        return self.state
    
    #%%
    def seed(self, seed=0):
        np.random.seed(seed)
