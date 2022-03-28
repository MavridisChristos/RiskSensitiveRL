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

import gym
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Hyper-parameters 

name = 'Online-Risk'

# Risk Sensitivity
risk_objective = 'beta'
risk_objective = 'betainverse'
risk_objective = 'sign'
# risk_objective = 'None'
risk_beta = -0.1
# risk = 0

# Online or Batch
online = True
# online = False
alpha = 1
update_buffer = 1001 # update actor-critic buffer for offline

# Baseline
baseline = True
baseline = False
value_approximation = True
value_approximation = False

# Simulation Epochs
train_loops = 11 # hundreds of training epochs
test_loops = 0 # hundreds of testing epochs
nepochs = 100
time_steps = 200 # time horizon for successful epoch

# Neural Networks
multiple_layers = False
nn_hidden_size = 256
lr = 0.001
gammaAC = 0.99

# Game
game='cartpole'
# game='acrobot'
# game='mountaincar'

# plot folder
dfolder = './plots'
os.makedirs(dfolder, exist_ok=True)

# save results file
results_folder = './results/'+game
os.makedirs(results_folder, exist_ok=True)
results_file = results_folder+'/'+name+'.pkl'
save_to_file = True
# save_to_file = False

# Animation
show_animation = False

# random seed
rs=0

#%% Environment Initialization and Random Seeds

if game=='cartpole':
    # import cartpole
    # env = cartpole.CartPoleEnv()
    env = gym.make("CartPole-v1")
elif game=='acrobot':
    # import acrobot
    # env = acrobot.AcrobotEnv()
    env = gym.make("Acrobot-v1")
elif game=='mountaincar':
    # import mountaincar
    # env = mountaincar.MountainCarEnv()
    env = gym.make("MountainCar-v0")
    
env_seed=env.seed(rs)
env.action_space.np_random.seed(rs)
np.random.seed(rs)
random.seed(rs) 
torch.manual_seed(rs)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#%% RL Model Initializtion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if multiple_layers:
    
    class Actor(nn.Module):
        def __init__(self, state_size, action_size, nn_hidden_size):
            super(Actor, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.linear1 = nn.Linear(self.state_size, 128)
            self.linear2 = nn.Linear(128, 256)
            self.linear3 = nn.Linear(256, self.action_size)
    
        def forward(self, state):
            output = F.relu(self.linear1(state))
            output = F.relu(self.linear2(output))
            output = self.linear3(output)
            distribution = Categorical(F.softmax(output, dim=-1))
            return distribution
    
    class Critic(nn.Module):
        def __init__(self, state_size, action_size, nn_hidden_size):
            super(Critic, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.linear1 = nn.Linear(self.state_size, 128)
            self.linear2 = nn.Linear(128, 256)
            self.linear3 = nn.Linear(256, 1)
    
        def forward(self, state):
            output = F.relu(self.linear1(state))
            output = F.relu(self.linear2(output))
            value = self.linear3(output)
            return value

else:
    
    class Actor(nn.Module):
        
        def __init__(self, state_size, action_size, nn_hidden_size):
            super(Actor, self).__init__()
    
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_size = nn_hidden_size
            self.actorlinear1 = nn.Linear(self.state_size, self.hidden_size)
            self.actorlinear2 = nn.Linear(self.hidden_size, self.action_size)
        
        def forward(self, state):
            
            # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            policy = self.actorlinear1(state)
            policy = F.relu(policy)
            policy = self.actorlinear2(policy)
            policy = F.softmax(policy, dim=-1)
            policy = Categorical(policy)
    
            return policy
        
    class Critic(nn.Module):
        
        def __init__(self, state_size, action_size, nn_hidden_size):
            super(Critic, self).__init__()
    
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_size = nn_hidden_size
            self.criticlinear1 = nn.Linear(self.state_size, self.hidden_size)
            self.criticlinear2 = nn.Linear(self.hidden_size, 1)
    
        def forward(self, state):
            
            # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            value = self.criticlinear1(state)
            value = F.relu(value)
            value = self.criticlinear2(value)
    
            return value
    
actor = Actor(state_size, action_size, nn_hidden_size)
a_optimizer = optim.Adam(actor.parameters(),lr=lr)
critic = Critic(state_size, action_size, nn_hidden_size)    
c_optimizer = optim.Adam(critic.parameters(),lr=lr)

#%% RL Model Update

def update_actor_critic(new_state,rewards,values,log_probs,perfect):
    
    log_probs = torch.cat(log_probs)
        
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gammaAC * R 
        returns.insert(0, R)
    returns = torch.cat(returns).detach()

    if baseline or value_approximation:
        values = torch.cat(values)
    else:
        values = 0
    if (baseline and perfect) or value_approximation:
        new_value = critic(new_state).detach()
    else:
        new_value = 0
        
    if risk_objective == 'beta':
        advantage = risk_beta*torch.exp(risk_beta*returns) + gammaAC * new_value - values
    elif risk_objective == 'betainverse':
        advantage = 1/risk_beta*torch.exp(risk_beta*returns) + gammaAC * new_value - values
    elif risk_objective == 'sign':
        advantage = np.sign(risk_beta)*torch.exp(risk_beta*returns) + gammaAC * new_value - values
    else:
        advantage = returns + gammaAC * new_value - values
        
    if value_approximation:
        actor_loss = -(log_probs * values.detach()).mean()
        sign = 1
        if risk_objective=='beta' or risk_objective=='betainverse' or risk_objective=='sign':
            sign = -np.sign(risk_beta)
        else:
            sign = -1
        critic_loss = sign*advantage.pow(2).mean()
    else:
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

    a_optimizer.zero_grad()
    actor_loss.backward()
    a_optimizer.step()
    
    if baseline or value_approximation:
        c_optimizer.zero_grad()
        critic_loss.backward()
        c_optimizer.step()
    
def online_update_actor_critic(new_state,rewards,values,log_probs,beta):
    
    returns = torch.cat(rewards).detach()
    log_probs = torch.cat(log_probs)
    values = torch.cat(values)
    new_value = critic(new_state).detach()
    
    if risk_objective == 'beta':
        advantage = risk_beta*torch.exp(risk_beta*returns) + gammaAC * new_value - values
    elif risk_objective == 'betainverse':
        advantage = 1/risk_beta*torch.exp(risk_beta*returns) + gammaAC * new_value - values
    elif risk_objective == 'sign':
        advantage = np.sign(risk_beta)*torch.exp(risk_beta*returns) + gammaAC * new_value - values
    else:
        advantage = returns + gammaAC * new_value - values
        
    actor_loss = -(log_probs * values.detach()).mean()
    sign = 1
    if risk_objective=='beta' or risk_objective=='betainverse' or risk_objective=='sign':
        sign = -np.sign(risk_beta)
    else:
        sign = -1
    critic_loss = sign*advantage.pow(2).mean()

    a_optimizer.param_groups[0]["lr"] = beta
    a_optimizer.zero_grad()
    actor_loss.backward()
    a_optimizer.step()
    
    c_optimizer.param_groups[0]["lr"] = beta
    c_optimizer.zero_grad()
    critic_loss.backward()
    c_optimizer.step()

#%% Training Loop

training_all=[]
training_avg=[]
training_std=[]
# for all training loops
for k in range(train_loops):
    avg = 0
    std2 = 0
    
    lr = lr * 0.95
        
    # for nepochs epochs (episodes)
    for i in range(nepochs):
        
        score = 0
        log_probs = []
        values = []
        rewards = []
        entropy = 0
        perfect=True
        
        # reset/observe current state
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        # repeat until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            value = critic(state)
            policy = actor(state)
            action = policy.sample()
            
            # observe new state
            new_state, reward, done, info = env.step(action.cpu().numpy())
            new_state = torch.FloatTensor(new_state).to(device) 
            
            # Update memory for RL model
            log_prob = policy.log_prob(action).unsqueeze(0)
            entropy += policy.entropy().mean()
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            log_probs.append(log_prob)
            score+= reward
            
            # Batch or Online Update
            if online:
                stepsize = lr*alpha/(t+1)
                online_update_actor_critic(new_state, rewards, values, log_probs, stepsize)
                log_probs = []
                values = []
                rewards = []
            else:
                if t>0 and t%update_buffer==0:
                    perfect = False
                    update_actor_critic(new_state, rewards, values, log_probs, perfect)
                    log_probs = []
                    values = []
                    rewards = []
                    
            state=new_state
            
            if done:
                perfect = False
                break
        
        # Episodic Update
        if len(rewards)>0:
            update_actor_critic(new_state, rewards, values, log_probs, perfect)
        
        # compute average over time_steps repeats 
        training_all.append(score)
        avg = avg + 1/(i+1) * (score-avg) # = avg * i/(i+1) + (t+1)/(i+1) 
        if i==0:
            std2 = (score - avg)**2 
        else:
            std2 = (i-1)/i * std2 + 1/(i+1) * (score - avg)**2             
        
    # Compute Average number of timesteps
    training_avg.append(avg)
    training_std.append(np.sqrt(std2))
    print(f'{k+1}-th episode: Average timesteps: {avg}')
    
#%% Save results to file 

if save_to_file:    

    my_results = [training_all, training_avg]
               
    if results_file != '':
        with open(results_file, mode='wb') as file:
            pickle.dump(my_results, file) 

#%% Plot Training Curve
    
fig = plt.figure(facecolor='white')

plt.title('Training Curve')
x=np.arange(len(training_all))+1
y=np.array(training_all)
plt.plot(x,y,color='b',alpha=0.1)
x=(np.arange(len(training_avg))+1)*nepochs 
x=np.insert(x, 0, 1)
y=np.array(training_avg)
y=np.insert(y,0,training_all[0])
plt.plot(x,y, label='Average Reward',color='b',linewidth=2)
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
plt.legend()
plt.show()
    
#%% Animation

if show_animation:
    
    time_steps = 1000
    avg = 0
    score = 0
    
    for i in range(1):
        
        state = env.reset()
            
        for t in range(time_steps):
            
            env.render()
        
            time.sleep(0.01)
                
            # pick next action
            value = critic(state)
            policy = actor(state)
            action = policy.sample()
            
            # observe new state
            new_state, reward, done, info = env.step(action.cpu().numpy())
            new_state = torch.FloatTensor(new_state).to(device) 
            score += reward
            
            state=new_state
            if done:
                print("Episode finished. Reward: {}".format(t+1))
                break
        
        avg = avg + 1/(i+1) * (score-avg)
        
    print(f'Average Reward: {avg}')
    env.close()
