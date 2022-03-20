#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Actor Critic with Gym OpenAI
    Christos Mavridis <mavridis@umd.edu>
    March 2022
"""

#%%

import os
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 

import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import cartpole

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Hyper-parameters 

# Low D or High D
# lowD = True
lowD = False

# Simulation Epochs
train_loops = 14 # hundreds of training epochs
test_loops = 3 # hundreds of testing epochs
nepochs = 100
time_steps = 200 # time horizon for successful epoch
online = True
update_buffer = 100 # update actor-critic buffer for offline

# plot folder
dfolder = './plots'
os.makedirs(dfolder, exist_ok=True)

# save results file
os.makedirs('./results', exist_ok=True)
results_file = './results/last_results.pkl'

# Plot
plot_figures = False

# Animation
show_animation = False

# Save results to file
save_to_file = False

# random seed
rs=1

#%% Actor-Critic Parameters 

nn_hidden_size = 256
learning_rate = 3e-4
gammaAC = 0.99

if False:
    gammaSA = 0.9 # RL discount
    epsilon = 0.2 # for epsilon-Greedy policy 
    epercent = 0.8 # epsilon = epercent * epsilon
    aa_init=0.1 # 0.9
    aa_step = 0.3 # 0.9

#%% Environment Initialization and Random Seeds

env = cartpole.CartPoleEnv()
env_seed=env.seed(rs)
env.action_space.np_random.seed(rs)
np.random.seed(rs)
random.seed(rs) 

#%% RL Model Initializtion

class ActorCritic(nn.Module):
    
    def __init__(self, num_inputs, num_actions, nn_hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, nn_hidden_size)
        self.critic_linear2 = nn.Linear(nn_hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, nn_hidden_size)
        self.actor_linear2 = nn.Linear(nn_hidden_size, num_actions)
    
    def forward(self, state):
        
        # critic: linear-relu-linear
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        # actor: linear-relu-linear-softmax
        policy = F.relu(self.actor_linear1(state))
        policy = F.softmax(self.actor_linear2(policy), dim=1)

        return value, policy

num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.n

if lowD:
    num_inputs = 2
    
actor_critic = ActorCritic(num_inputs, num_outputs, nn_hidden_size)
ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

def update_actor_critic(new_state,rewards,values,log_probs):
    
    Qval, _ = actor_critic.forward(new_state)
    Qval = Qval.detach().numpy()[0,0]
    # all_rewards.append(np.sum(rewards))
    
    # compute Q values
    Qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + gammaAC * Qval
        Qvals[t] = Qval

    #update actor critic
    values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals)
    log_probs = torch.stack(log_probs)
    
    advantage = Qvals - values
    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    ac_loss = actor_loss + critic_loss

    ac_optimizer.zero_grad()
    ac_loss.backward()
    ac_optimizer.step()
    
    
def online_update_actor_critic(new_state,rewards,values,log_probs,beta):
    
    reward = torch.FloatTensor(rewards)
    value = torch.FloatTensor(values)
    log_prob = torch.stack(log_probs)
    
    Qval, _ = actor_critic.forward(new_state)
    
    advantage = -(torch.exp(reward) + gammaAC * Qval - value)
    critic_loss = beta * 0.5 * advantage.pow(2).mean()
    actor_loss = beta * log_prob * advantage
    ac_loss = actor_loss + critic_loss

    ac_optimizer.zero_grad()
    ac_loss.backward()
    ac_optimizer.step()

#%% Plot Initial State Space

def vis_V(actor_critic):
    
    three_D = False
    
    # defining x, y, z co-ordinates 
    mesh_points = 101
    box2 = [-2.5,2.5]
    box3 = [-2.5,2.5]
    s2 = np.linspace(-box2, box2, mesh_points)
    s3 = np.linspace(-box3, box3, mesh_points)
    S2, S3 = np.meshgrid(s2, s3)
    
    # Plot the surface.
    V_plot = np.zeros((mesh_points,mesh_points))
    for i in range(mesh_points):
        for j in range(mesh_points):
            state = np.array([s2[i],s3[j]])
            V_plot[i,j],_ = actor_critic.forward(state)
      
    if three_D:
        # create a new figure for plotting 
        fig, ax = plt.subplots(facecolor='white',subplot_kw={"projection": "3d"})
        
        surf = ax.plot_surface(S2, S3, V_plot, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        # Customize x,y axis
        ax.tick_params(axis='both', which='major', labelsize=8)
        # ax.set_xlim(-box2, box2)
        ax.set_xlabel(r'$\theta$')
        # ax.set_ylim(-box3, box3)
        ax.set_ylabel(r'$\dot\theta$')
        ax.set_zlabel(r'$V(\theta,\dot\theta)$')
        
        ax.view_init(50, 50)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.3, aspect=5, pad=0.0)

    else:
        
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(V_plot,extent=[-1,1,-1,1],cmap=cm.coolwarm)
        x_label_list = [-2.5, 0, 2.5]
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(x_label_list)
        y_label_list = [-2.5, 0, 2.5]
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(y_label_list)
        fig.colorbar(img,shrink=0.3, aspect=5, pad=0.01)
    
if lowD and plot_figures:    
    vis_V(actor_critic=actor_critic)
    plt.show()
    
#%% Training Loop

training_avg=[]
# for all training loops
for k in range(train_loops):
    avg = 0
    # for nepochs epochs (episodes)
    for i in range(nepochs):
        
        beta = 1
        log_probs = []
        values = []
        rewards = []
        
        # reset/observe current state
        state = env.reset()
        if lowD:
            state = state[2:]
            
        # repeat until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            value, policy = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            policy_np = policy.detach().numpy() 
            action = np.random.choice(num_outputs, p=np.squeeze(policy_np))
            log_prob = torch.log(policy.squeeze(0)[action])
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if lowD:
                new_state = new_state[2:]
            
            # Update memory for RL model
            rewards.append(cost)
            values.append(value)
            log_probs.append(log_prob)
            
            if online:
                online_update_actor_critic(new_state, rewards, values, log_probs, beta)
                beta = beta*gammaAC
                log_probs = []
                values = []
                rewards = []
            else:
                if t%update_buffer==0:
                    update_actor_critic(new_state, rewards, values, log_probs)
                    log_probs = []
                    values = []
                    rewards = []
                    
            state=new_state
            if terminate:
                # print("Episode finished after {} timesteps".format(t+1))
                if len(rewards)>0:
                    update_actor_critic(new_state, rewards, values, log_probs)
                break
        
        # compute average over time_steps repeats    
        avg = avg + 1/(i+1) * (t+1-avg) # = avg * i/(i+1) + (t+1)/(i+1)    
                    
    # Compute Average number of timesteps
    training_avg.append(avg)
    print(f'{k+1}-th episode: Average timesteps: {avg}')
        
    # Visualize Value Function
    if lowD and plot_figures:    
        vis_V(actor_critic=actor_critic)
        plt.show()

#%% Testing Loop

time_steps=1000    
testing_avg=[]
avg = 0

for k in range(test_loops):

    for i in range(nepochs):
        
        # reset/observe current state
        state = env.reset()
        if lowD:
            state = state[2:]
        # repeat until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            value, policy = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            policy_np = policy.detach().numpy() 
            # action = np.random.choice(num_outputs, p=np.squeeze(policy_np))
            action = np.argmax(np.squeeze(policy_np))
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if lowD:
                new_state = new_state[2:]
            
            state=new_state
            if terminate:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        
        # compute average over 100 repeats    
        avg = avg + 1/(i+1) * (t+1-avg) # = avg * i/(i+1) + (t+1)/(i+1)   
        
    update_once=False
    testing_avg.append(avg)
    print(f'Testing: Average timesteps: {avg}.')
    
#%% Save results to file 

if save_to_file:    

    my_results = [training_avg, testing_avg]
               
    if results_file != '':
        with open(results_file, mode='wb') as file:
            pickle.dump(my_results, file) 

#%% Plot Training Curve
    
fig = plt.figure(facecolor='white')

plt.title('Training Curve')
plt.plot(np.arange(len(training_avg))+1,training_avg, label='Training Averages')
plt.plot(len(training_avg)+np.zeros(len(testing_avg))+1,testing_avg,'r*',
                  label='Testing Averages')
plt.xlabel('Hundreds of episodes')
plt.ylabel('Average number of timesteps')
plt.legend()
plt.show()
    
#%% Animation

if show_animation:
    
    time_steps = 1000
    avg = 0
    
    for i in range(1):
        
        state = env.reset()
        if lowD:
            state = state[2:]
            
        for t in range(time_steps):
            
            env.render()
        
            time.sleep(0.01)
                
            # pick next action
            value, policy = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            policy_np = policy.detach().numpy() 
            # action = np.random.choice(num_outputs, p=np.squeeze(policy_np))
            action = np.argmax(np.squeeze(policy_np))
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if lowD:
                new_state = new_state[2:]
            
            state=new_state
            if terminate:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        avg = avg + 1/(i+1) * (t+1-avg)
        
    print(f'Average Number of timesteps: {avg}')
    env.close()
