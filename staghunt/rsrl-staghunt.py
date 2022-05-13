#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Risk-Sensitive Reinforcement Learning for Stag-Hunt Game
    Erfaun Noorani, Mavridis Christos, John S. Baras 
    Department of Electrical and Computer Engineering 
    University of Maryland
    <enoorani,mavridis,baras>@umd.edu
    May 2022
"""

#%% Importing Modules

"""
    Dependencies: PyTorch (Neural Network Training), Stag-Hunt model
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import gym
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# import CTM
import staghunt

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')
    

#%% Hyper-Parameters

name = ''
model_var = [0] 
risk_beta = -0.1 #-0.1
risk_objective = 'BETA'
train_loops = 1
test_loops = 0
nepochs = 50
time_steps = 200
gammaAC = 0.99
nn_hidden_size = [16]
lr = 0.01
cut_lr=False
look_ahead = 1
baseline = True
a_outer = 0.0
a_inner = 0.0 
rs=0

#%% Environment Initialization and Random Seeds

game = 'staghunt'
env = staghunt.staghunt()

state_size = env.state_size
action_size = env.action_size

# Fix random seeds
env.seed(rs)
np.random.seed(rs)
random.seed(rs) 
torch.manual_seed(rs)

# Results File Name
results_folder = './results/'+game
os.makedirs(results_folder, exist_ok=True)

def n2t(x,d=2):
    text=''
    if x<0:
        x= -x
        text+='m'
    if d==1:
        text+=f'{int(x):01}'
    elif d==2:
        text+=f'{int(x):02}'
    elif d==3:
        text+=f'{int(x):03}'  
    elif d==4:
        text+=f'{int(x):04}'  
    return text

def b2t(b,text='bl'):
    if b:
        return text
    else:
        return ''
 
if name=='':
    name = 'LA'+n2t(look_ahead)+b2t(baseline,'BL')+'-'+risk_objective+n2t(risk_beta*100,2)+'-NN'+n2t(nn_hidden_size[0],3)+'/'+ \
            'LR'+b2t(cut_lr,'CUT')+n2t(lr*10000,4)+'-Ao'+n2t(a_outer*100,2)+'-Ai'+n2t(a_inner*100,2)
    os.makedirs(results_folder+'/'+'LA'+n2t(look_ahead)+b2t(baseline,'BL')+'-'+risk_objective+n2t(risk_beta*100,2)+'-NN'+n2t(nn_hidden_size[0],3), exist_ok=True)

#%% RL Model Initializtion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if len(nn_hidden_size)>1:
    
    class Actor(nn.Module):
        def __init__(self, state_size, action_size, nn_hidden_size):
            super(Actor, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_size = nn_hidden_size
            self.linear1 = nn.Linear(self.state_size, self.hidden_size[0])
            self.linear2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
            self.linear3 = nn.Linear(self.hidden_size[1], self.action_size)
    
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
            self.hidden_size = nn_hidden_size
            self.linear1 = nn.Linear(self.state_size, self.hidden_size[0])
            self.linear2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
            self.linear3 = nn.Linear(self.hidden_size[1], 1)
    
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
            self.actorlinear1 = nn.Linear(self.state_size, self.hidden_size[0])
            self.actorlinear2 = nn.Linear(self.hidden_size[0], self.action_size)
        
        def forward(self, state):
            
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
            self.criticlinear1 = nn.Linear(self.state_size, self.hidden_size[0])
            self.criticlinear2 = nn.Linear(self.hidden_size[0], 1)
    
        def forward(self, state):
            
            value = self.criticlinear1(state)
            value = F.relu(value)
            value = self.criticlinear2(value)
    
            return value

def stepsize(lr,n,outer=False):
    if outer:
        if n==1:
            a= lr 
        else:
            a= lr * 1/(1 + n*a_outer)
    else:
        a= lr * 1/(1 + n*a_inner)
    return a

actor1 = Actor(state_size, action_size, nn_hidden_size)
critic1 = Critic(state_size, action_size, nn_hidden_size)    
actor2 = Actor(state_size, action_size, nn_hidden_size)
critic2 = Critic(state_size, action_size, nn_hidden_size)    
a1_optimizer = optim.Adam(actor1.parameters(),lr=lr)
c1_optimizer = optim.Adam(critic1.parameters(),lr=lr)
a2_optimizer = optim.Adam(actor2.parameters(),lr=lr)
c2_optimizer = optim.Adam(critic2.parameters(),lr=lr)

#%% RL Model Update
    
def update_actor_critic(new_state,rewards1,rewards2,values1,values2,log_probs1,log_probs2,beta,baseline):
    
    # Agent 1
    
    R = 0
    returns = []
    for step in reversed(range(len(rewards1))):
        R = rewards1[step] + gammaAC * R 
        returns.insert(0, R)
    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs1)
    values = torch.cat(values1)
    new_value = critic1(new_state).detach()
    
    if risk_beta!=0:
        if risk_objective=='BETA':
            advantage = risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAI':
            advantage = 1/risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAS':
            advantage = np.sign(risk_beta)*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        else:
            advantage = (returns + gammaAC * new_value) - values
    else:
        advantage = (returns + gammaAC * new_value) - values
    
    if risk_beta!=0:
        sign = -np.sign(risk_beta)
    else:
        sign = -1
        
    actor_loss = -(log_probs * values.detach()).mean() 
    
    # actor_loss = -(log_probs * values.detach()).mean() 
    a1_optimizer.param_groups[0]["lr"] = beta
    a1_optimizer.zero_grad()
    actor_loss.backward()
    a1_optimizer.step()
    
    critic_loss = sign*advantage.pow(2).mean()
    c1_optimizer.param_groups[0]["lr"] = beta
    c1_optimizer.zero_grad()
    critic_loss.backward()
    c1_optimizer.step()
    
    # Agent 2
    
    R = 0
    returns = []
    for step in reversed(range(len(rewards2))):
        R = rewards2[step] + gammaAC * R 
        returns.insert(0, R)
    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs2)
    values = torch.cat(values2)
    new_value = critic2(new_state).detach()
    
    if risk_beta!=0:
        if risk_objective=='BETA':
            advantage = risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAI':
            advantage = 1/risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAS':
            advantage = np.sign(risk_beta)*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        else:
            advantage = (returns + gammaAC * new_value) - values
    else:
        advantage = (returns + gammaAC * new_value) - values
    
    if risk_beta!=0:
        sign = -np.sign(risk_beta)
    else:
        sign = -1
        
    actor_loss = -(log_probs * values.detach()).mean() 
    
    # actor_loss = -(log_probs * values.detach()).mean() 
    a2_optimizer.param_groups[0]["lr"] = beta
    a2_optimizer.zero_grad()
    actor_loss.backward()
    a2_optimizer.step()
    
    critic_loss = sign*advantage.pow(2).mean()
    c2_optimizer.param_groups[0]["lr"] = beta
    c2_optimizer.zero_grad()
    critic_loss.backward()
    c2_optimizer.step()
        
def update_reinforce(new_state,rewards1,rewards2,values1,values2,log_probs1,log_probs2,beta,baseline):
    
    # Agent 1
    
    R = 0
    returns = []
    for step in reversed(range(len(rewards1))):
        R = rewards1[step] + gammaAC * R 
        returns.insert(0, R)
    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs1)
    values = torch.cat(values1)
    new_value = critic1(new_state).detach()
    
    if not baseline:
        values = 0*values
    
    if risk_beta!=0:
        if risk_objective=='BETA':
            advantage = risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAI':
            advantage = 1/risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAS':
            advantage = np.sign(risk_beta)*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        else:
            advantage = (returns + gammaAC * new_value) - values
    else:
        advantage = (returns + gammaAC * new_value) - values
    
    if risk_beta!=0:
        sign = +1 #-np.sign(risk_beta)
    else:
        sign = -1
        
    actor_loss = -(log_probs * advantage.detach()).mean() 
    a1_optimizer.param_groups[0]["lr"] = beta
    a1_optimizer.zero_grad()
    actor_loss.backward()
    a1_optimizer.step()
    
    if baseline:
        critic_loss = sign*advantage.pow(2).mean()
        c1_optimizer.param_groups[0]["lr"] = beta
        c1_optimizer.zero_grad()
        critic_loss.backward()
        c1_optimizer.step()

    # Agent 2
    
    R = 0
    returns = []
    for step in reversed(range(len(rewards2))):
        R = rewards2[step] + gammaAC * R 
        returns.insert(0, R)
    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs2)
    values = torch.cat(values2)
    new_value = critic2(new_state).detach()
    
    if not baseline:
        values = 0*values
    
    if risk_beta!=0:
        if risk_objective=='BETA':
            advantage = risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAI':
            advantage = 1/risk_beta*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        elif risk_objective=='BETAS':
            advantage = np.sign(risk_beta)*torch.exp(risk_beta * (returns + gammaAC * new_value)) - values
        else:
            advantage = (returns + gammaAC * new_value) - values
    else:
        advantage = (returns + gammaAC * new_value) - values
    
    if risk_beta!=0:
        sign = +1 #-np.sign(risk_beta)
    else:
        sign = -1
        
    actor_loss = -(log_probs * advantage.detach()).mean() 
    a2_optimizer.param_groups[0]["lr"] = beta
    a2_optimizer.zero_grad()
    actor_loss.backward()
    a2_optimizer.step()
    
    if baseline:
        critic_loss = sign*advantage.pow(2).mean()
        c2_optimizer.param_groups[0]["lr"] = beta
        c2_optimizer.zero_grad()
        critic_loss.backward()
        c2_optimizer.step()

#%% Training Loop

training_all=[]
training_all1=[]
training_all2=[]
print('*** Training ***')

# for all training loops
for k in range(train_loops):
    
    lr = stepsize(lr, k+1, outer=True)
    
    # for nepochs epochs (episodes)
    for i in range(nepochs):
        
        score1 = 0
        log_probs1 = []
        values1 = []
        rewards1 = []
        score2 = 0
        log_probs2 = []
        values2 = []
        rewards2 = []
        
        nash = 0
        
        # reset/observe current state
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
    
        # repeat until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            value1 = critic1(state)
            policy1 = actor1(state)
            action1 = policy1.sample()
            value2 = critic2(state)
            policy2 = actor2(state)
            action2 = policy2.sample()
            
            # observe new state
            new_state,reward1,reward2,done = env.step(action1.cpu().numpy(),action2.cpu().numpy())
            new_state = torch.FloatTensor(new_state).to(device) 
            
            # Update memory for RL model
            log_prob1 = policy1.log_prob(action1).unsqueeze(0)
            values1.append(value1)
            rewards1.append(torch.tensor([reward1], dtype=torch.float, device=device))
            log_probs1.append(log_prob1)
            score1+= reward1
            log_prob2 = policy2.log_prob(action2).unsqueeze(0)
            values2.append(value2)
            rewards2.append(torch.tensor([reward2], dtype=torch.float, device=device))
            log_probs2.append(log_prob2)
            score2+= reward2
            
            # Batch or Online Update
            if t>0 and look_ahead>0 and t%look_ahead==0:
                update_actor_critic(new_state,rewards1,rewards2,
                                    values1,values2,log_probs1,log_probs2,
                                    stepsize(lr,int((t+1)/look_ahead)), baseline)
                log_probs1 = []
                values1 = []
                rewards1 = []
                log_probs2 = []
                values2 = []
                rewards2 = []
                    
            state=new_state
            
            success = 1 if reward1+reward2==10 else 0
            nash += success
            
            if done:
                print("Episode Failed")
                break
        
        if len(rewards1)>0 and look_ahead>0:
            update_actor_critic(new_state,rewards1,rewards2,
                                values1,values2,log_probs1,log_probs2,
                                stepsize(lr,int((t+1)/look_ahead)), baseline)
        elif len(rewards1)>0 and look_ahead==0:
            update_reinforce(new_state,rewards1,rewards2,
                                values1,values2,log_probs1,log_probs2,
                                stepsize(lr,0), baseline)

        # compute average over time_steps repeats 
        training_all1.append(score1)
        training_all2.append(score2)
        training_all.append(nash/time_steps)
        
    # print(f'{(k+1)*nepochs}: Training Average: {avg:.2f} +- {np.sqrt(std2):.2f}')
    
#%% Plot Training Curve

# colors = ['r','g','m','y','k','c','pink']
colors = mcolors.TABLEAU_COLORS
colors = list(colors.keys())

fig = plt.figure(facecolor='white')

plt.title(name)

# x=np.arange(len(training_all))+1
# y=np.array(training_all)
# plt.plot(x,y,color='b',alpha=0.05)

x=np.arange(len(training_all))
y=np.array(training_all)
plt.plot(x,y, label='Training',color='b',marker='*',linewidth=3,markersize=12)
# x=np.arange(len(training_all1))
# y=np.array(training_all1)
# plt.plot(x,y, label='Training',color='b',linewidth=2)
# x=np.arange(len(training_all2))
# y=np.array(training_all2)
# plt.plot(x,y, label='Training',color='r',linewidth=2)

plt.xlabel('Number of episodes')
plt.ylabel('Nash Frequency')

plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
plt.legend()

plot_file = results_folder+'/'+name+'.png'
fig.savefig(plot_file, format = 'png')  

plt.show()
    
#%% Save results to file 
    
my_results = [training_all1,training_all2]
       
results_file = results_folder+'/'+name+'.pkl'
with open(results_file, mode='wb') as file:
    pickle.dump(my_results, file) 
    