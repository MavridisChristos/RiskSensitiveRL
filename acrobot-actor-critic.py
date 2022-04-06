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

import gym
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Hyper-parameters 

name = 'test'

# Risk Sensitivity
risk_objective = 'beta'
# risk_objective = 'betainverse'
# risk_objective = 'sign'
# risk_objective = 'None'
risk_beta = -0.01
# risk = 0

# Stepsizes
lr = 0.0005
a_outer = 0.0 # 0 means constant
a_inner = 0.0 # 0 means constant
cut_lr=True # reduces lr if reward>-100
def stepsize(lr,n,outer=False):
    
    if outer:
        if n==1:
            a= lr 
        else:
            a= lr * 1/(1 + n*a_outer)
    else:
        a= lr * 1/(1 + n*a_inner)
        # if n==1:
            # a= lr * 1/(1 + n*0.9)
        # else:
            # a= lr * alpha/(alpha_c + n*np.log(n))
    
    return a

# Simulation Epochs
train_loops = 100 # hundreds of training epochs
test_loops = 20 # hundreds of testing epochs
nepochs = 10
time_steps = 500 # time horizon for successful epoch
gammaAC = 0.99

# Neural Networks
nn_hidden_size = 16
multiple_layers = True
multiple_layers = False

# Game
# game='cartpole'
game='acrobot'
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
    import cartpole
    env = cartpole.CartPoleEnv()
    # env = gym.make("CartPole-v0")
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
            self.linear1 = nn.Linear(self.state_size, 32)
            self.linear2 = nn.Linear(32, 64)
            self.linear3 = nn.Linear(64, self.action_size)
    
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
            self.linear1 = nn.Linear(self.state_size, 32)
            self.linear2 = nn.Linear(32, 64)
            self.linear3 = nn.Linear(64, 1)
    
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
    
def online_update_actor_critic(new_state,rewards,values,log_probs,beta):
    
    returns = torch.cat(rewards).detach()
    log_probs = torch.cat(log_probs)
    values = torch.cat(values)
    new_value = critic(new_state).detach()
    
    if risk_objective == 'beta':
        advantage = risk_beta*torch.exp(risk_beta*(returns+ gammaAC * new_value)) - values
    elif risk_objective == 'betainverse':
        advantage = 1/risk_beta*torch.exp(risk_beta*(returns+ gammaAC * new_value)) - values
    elif risk_objective == 'sign':
        # advantage = np.sign(risk_beta)*torch.exp(risk_beta*(returns)) + gammaAC * new_value - values
        advantage = np.sign(risk_beta)*torch.exp(risk_beta*(returns+ gammaAC * new_value)) - values
    else:
        advantage = returns + gammaAC * new_value - values
        
    # actor_loss = -(log_probs * advantage.detach()).mean()
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
print('*** Training ***')

# for all training loops
for k in range(train_loops):
    
    avg = 0
    std2 = 0
    
    lr = stepsize(lr, k+1, outer=True)
    
    # for nepochs epochs (episodes)
    for i in range(nepochs):
        
        score = 0
        log_probs = []
        values = []
        rewards = []
        
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
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            log_probs.append(log_prob)
            score+= reward
            
            # Batch or Online Update
            online_update_actor_critic(new_state, rewards, values, log_probs, stepsize(lr,t+1))
            log_probs = []
            values = []
            rewards = []
                    
            state=new_state
            
            if done:
                break
        
        # compute average over time_steps repeats 
        training_all.append(score)
        avg = avg + 1/(i+1) * (score-avg) # = avg * i/(i+1) + (t+1)/(i+1) 
        if i==0:
            std2 = (score - avg)**2 
        else:
            std2 = (i-1)/i * std2 + 1/(i+1) * (score - avg)**2             
        
    if cut_lr and avg>-100:
        lr = 0.01*lr
        
    # Compute Average number of timesteps
    training_avg.append(avg)
    training_std.append(np.sqrt(std2))
    print(f'{(k+1)*nepochs}: Training Average: {avg:.2f} +- {np.sqrt(std2):.2f}')

#%% Testing Loop

testing_all=[]
testing_avg=[]
testing_std=[]
print('*** Testing ***')

# for all testing loops
for k in range(test_loops):
    avg = 0
    std2 = 0
    
    # for nepochs epochs (episodes)
    for i in range(nepochs):
        
        score = 0
        
        # reset/observe current state
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        # repeat until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            policy = actor(state)
            action = policy.sample()
            
            # observe new state
            new_state, reward, done, info = env.step(action.cpu().numpy())
            new_state = torch.FloatTensor(new_state).to(device) 
            score += reward
            
            # New State
            state=new_state
            
            if done:
                break
        
        # compute average over time_steps repeats 
        testing_all.append(score)
        avg = avg + 1/(i+1) * (score-avg) # = avg * i/(i+1) + (t+1)/(i+1) 
        if i==0:
            std2 = (score - avg)**2 
        else:
            std2 = (i-1)/i * std2 + 1/(i+1) * (score - avg)**2             
        
    # Compute Average number of timesteps
    testing_avg.append(avg)
    testing_std.append(np.sqrt(std2))
    print(f'{(k+1)*nepochs}: Testing Average: {avg:.2f} +- {np.sqrt(std2):.2f}')

#%% Plot Training Curve

fig = plt.figure(facecolor='white')

plt.title('Training Curve')

x=np.arange(len(training_all))+1
y=np.array(training_all)
plt.plot(x,y,color='b',alpha=0.05)

x=(np.arange(len(training_avg))+1)*nepochs 
x=np.insert(x, 0, 1)
y=np.array(training_avg)
y=np.insert(y,0,training_all[0])
sigma = np.array(training_std)
sigma = np.insert(sigma,0,sigma[0])
xfill = np.concatenate([x, x[::-1]])
yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
yfill = np.maximum(yfill,min(training_all)*np.ones_like(yfill))
yfill = np.minimum(yfill,max(training_all)*np.ones_like(yfill))
plt.plot(x,y, label='Training Average',color='b',linewidth=2)
plt.fill(xfill,yfill,
          alpha=.1, fc='b', ec='None')


x=len(training_all)+np.arange(len(testing_all))+1
y=np.array(testing_all)
plt.plot(x,y,color='r',alpha=0.05)

x=(np.arange(len(testing_avg))+1)*nepochs + len(training_all)
x=np.insert(x, 0, len(training_all))
y=np.array(testing_avg)
y=np.insert(y,0,testing_avg[0])
sigma = np.array(testing_std)
sigma = np.insert(sigma,0,sigma[0])
xfill = np.concatenate([x, x[::-1]])
yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
yfill = np.maximum(yfill,min(testing_all)*np.ones_like(yfill))
yfill = np.minimum(yfill,max(testing_all)*np.ones_like(yfill))
plt.plot(x,y, label='Testing Average',color='r',linewidth=2)
plt.fill(xfill,yfill,
          alpha=.1, fc='r', ec='None')

plt.xlabel('Number of episodes')
plt.ylabel('Reward')

plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
plt.legend()
plt.show()

#%% Save results to file 

if save_to_file:    

    my_results = [training_all, training_avg, training_std, testing_all, testing_avg, testing_std]
               
    if results_file != '':
        with open(results_file, mode='wb') as file:
            pickle.dump(my_results, file) 
    
#%% Animation

if show_animation:
    
    time_steps = 1000
    avg = 0
    score = 0
    
    for i in range(1):
        
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        for t in range(time_steps):
            
            env.render()
        
            time.sleep(0.01)
                
            # pick next action
            policy = actor(state)
            action = policy.sample()
            
            # observe new state
            new_state, reward, done, info = env.step(action.cpu().numpy())
            new_state = torch.FloatTensor(new_state).to(device) 
            score += reward
            
            # New State
            state=new_state
            
            if done:
                print("Episode finished. Reward: {}".format(score))
                break
        
        avg = avg + 1/(i+1) * (score-avg)
        
    print(f'Average Reward: {avg}')
    env.close()
