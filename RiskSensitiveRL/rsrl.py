#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Policy Gradient Algorithms (from REINFORCE to Actor-Critic)
        for Risk-Sensitive Reinforcement Learning
    
    Christos Mavridis, Erfaun Noorani, John Baras <{mavridis,enoorani,baras}@umd.edu>
    Department of Electrical and Computer Engineering 
    University of Maryland
"""

#%% Importing Modules

"""
    Dependencies: PyTorch (Neural Network Training), Gym (RL environments)
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import gym
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Hyper-parameters 

'''

### Project Name

- name = '' 
    # For saving model and results
    # If empty, project name includes all hyper-parameters used


### Game

- game='cartpole'
    # Any Gym environment
    # Supported here: 'cartpole', 'acrobot'


### Risk-Sensitive Parameters

- risk_beta = -0.1
    # J = risk_beta * exp( risk_beta * R )
    # 0 corresponds to the risk-neutral case: J = R
    # Typical values: [-0.5, -0.1, -0.01, 0 , 0.01, 0.1, 0.5]

- risk_objective = 'BETA' 
    # 'BETA': J = risk_beta * exp( risk_beta * R )
    # 'BETAI': J = 1/risk_beta * exp( risk_beta * R )
    # 'BETAS': J = sgn(risk_beta) * exp( risk_beta * R )

- gammaAC = 0.99
    # R = \sum_t gammaAC**t r_t    
    # diminishing returns: gammaAC in [0,1)


### REINFORCE vs Actor-Critic

- look_ahead = 0
    # 0: REINFROCE
    # 1: TD Actor-Critic
    # >1: Batch Actor-Critic

- baseline = False
    # Include Baseline in REINFORCE
    # Baseline is computed using a critic NN


### Training Loops

- train_loops = 100
    # Number of train loops
    # Increase train_loops to decrease nepochs as discussed bellow

- test_loops = 50
    # Number of test loops

- nepochs = 10
    # Number of epochs in each train loop
    # Controls the batch size over which the performance statistics are computed

- time_steps = 200 
    # Number of maximum timesteps (when no failure) in each epoch
    
    # for k in range(train_loops):
    #     for i in range(nepochs):
    #         for t in range(time_steps):
    #             observe
    #             act
    #             (update actor-critic)
    #         (update reinforce) 
    #         gather statistics (average score, variance, etc.) for every epoch 
    
### Neural Networks and Learning Rate

- nn_actor = [16]
    # Number of neurons in the actor neural network
    # Supports up to two layers, e.g., [32,16]

- nn_critic = [16]
    # Number of neurons in the critic neural network
    # Supports up to two layers, e.g., [32,16]
    
- lr = 0.01
    # Learning rate for Adam optimizer
    # Very important parameter for convergence: high sensitivity  

- cut_lr = False
    # mannually decrease learning rate if desired average score is reached
    # (developer mode)

- a_inner = 0.0 
    # Update learning rate according to stochastic approximation theory
    # lr = lr * 1/(1 + t * a_inner) where t is the timestep
    # 0 for no decreasing

- a_outer = 0.0
    # Update learning rate according to stochastic approximation theory
    # lr = lr * 1/(1 + k * a_outer) where k is the train loop
    # 0 for no decreasing

### Model Variations

- model_var = [round(0.2*(i+1),1) for i in range(10)]
    # Testing in environments with different model parameters
    # The changing parameter is pre-specified for each environment but can change (see Testing Loops) 
    # Nominal values: length=0.5 for cartpole, LINK_LENGTH_1=1.0 for acrobot
    
### Random Seed

- rs = 0
    # Random seed used for everything (Gym environment, random, np.random, and torch (NNs))
    # (developer mode)

'''

#%%

def train(
        # Name and Game
        name = '', 
        game='cartpole',
        # REINFORCE or Actor-Critic 
        look_ahead = 0,
        baseline = False,
        # Risk-Sensitivity
        risk_beta = -0.1,
        risk_objective = 'BETA', 
        gammaAC = 0.99,
        # Training Loops
        train_loops = 100,
        test_loops = 50,
        nepochs = 10,
        time_steps = 200, 
        # Neural Networks and Learning Rate
        nn_actor = [16],
        nn_critic = [16],
        lr = 0.01,
        a_outer = 0.0,
        a_inner = 0.0, 
        cut_lr=False,
        # Model Variations
        model_var = [round(0.2*(i+1),1) for i in range(10)],
        rs=0):
    
    #%% Environment Initialization and Random Seeds
    
    if game=='cartpole':
        gym.envs.register(
            id='CartPoleExtraLong-v0',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            max_episode_steps=time_steps,
            )
        env = gym.make('CartPoleExtraLong-v0')
    elif game=='acrobot':
        # env = gym.make("Acrobot-v1")
        gym.envs.register(
            id='AcrobotExtraLong-v0',
            entry_point='gym.envs.classic_control:AcrobotEnv',
            max_episode_steps=time_steps
            )
        env = gym.make('AcrobotExtraLong-v0')
    
    def goal(avg):
        if game=='cartpole':
            return avg>199
        elif game=='acrobot':
            return avg>-100
        
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Fix random seeds
    _=env.seed(rs)
    env.action_space.np_random.seed(rs)
    np.random.seed(rs)
    random.seed(rs) 
    torch.manual_seed(rs)
    
    # Results File Name
    results_folder = '../results/'+game
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
        name = 'LA'+n2t(look_ahead)+b2t(baseline,'BL')+'-'+risk_objective+n2t(risk_beta*100,2)+'-NN'+n2t(nn_actor[0],2)+n2t(nn_critic[0],2)+'/'+ \
                'LR'+b2t(cut_lr,'CUT')+n2t(lr*10000,4)+'-Ao'+n2t(a_outer*100,2)+'-Ai'+n2t(a_inner*100,2)
        os.makedirs(results_folder+'/'+'LA'+n2t(look_ahead)+b2t(baseline,'BL')+'-'+risk_objective+n2t(risk_beta*100,2)+'-NN'+n2t(nn_actor[0],2)+n2t(nn_critic[0],2), exist_ok=True)
    
    #%% RL Model Initializtion
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Actor
    
    if len(nn_actor)>1:
        
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
            
    # Critic
    
    if len(nn_critic)>1:
        
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
    
    actor = Actor(state_size, action_size, nn_actor)
    critic = Critic(state_size, action_size, nn_critic)    
    a_optimizer = optim.Adam(actor.parameters(),lr=lr)
    c_optimizer = optim.Adam(critic.parameters(),lr=lr)
    
    
    #%% RL Model Update
        
    def update_actor_critic(new_state,rewards,values,log_probs,beta,baseline):
        
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gammaAC * R 
            returns.insert(0, R)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        new_value = critic(new_state).detach()
        
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
        a_optimizer.param_groups[0]["lr"] = beta
        a_optimizer.zero_grad()
        actor_loss.backward()
        a_optimizer.step()
        
        critic_loss = sign*advantage.pow(2).mean()
        c_optimizer.param_groups[0]["lr"] = beta
        c_optimizer.zero_grad()
        critic_loss.backward()
        c_optimizer.step()
            
    def update_reinforce(new_state,rewards,values,log_probs,beta,baseline):
        
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gammaAC * R 
            returns.insert(0, R)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        new_value = critic(new_state).detach()
        
        if not baseline:
            values = 0*values
            new_value = 0*new_value
            
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
        a_optimizer.param_groups[0]["lr"] = beta
        a_optimizer.zero_grad()
        actor_loss.backward()
        a_optimizer.step()
        
        if baseline:
            critic_loss = sign*advantage.pow(2).mean()
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
                if t>0 and look_ahead>0 and t%look_ahead==0:
                    update_actor_critic(new_state, rewards, values, log_probs, stepsize(lr,int((t+1)/look_ahead)), baseline)
                    log_probs = []
                    values = []
                    rewards = []
                        
                state=new_state
                
                if done:
                    break
            
            if len(rewards)>0 and look_ahead>0:
                update_actor_critic(new_state, rewards, values, log_probs, stepsize(lr,int((t+1)/look_ahead)), baseline)
            elif len(rewards)>0 and look_ahead==0:
                update_reinforce(new_state, rewards, values, log_probs, stepsize(lr,0), baseline)
    
            # compute average over time_steps repeats 
            training_all.append(score)
            avg = avg + 1/(i+1) * (score-avg) # = avg * i/(i+1) + (t+1)/(i+1) 
            if i==0:
                std2 = (score - avg)**2 
            else:
                std2 = (i-1)/i * std2 + 1/(i+1) * (score - avg)**2             
            
        if cut_lr and goal(avg):
            lr = 0.01*lr
            
        # Compute Average number of timesteps
        training_avg.append(avg)
        training_std.append(np.sqrt(std2))
        print(f'{(k+1)*nepochs}: Training Average: {avg:.2f} +- {np.sqrt(std2):.2f}')
    
    #%% Testing Loop
    
    testing_all=[]
    testing_avg=[]
    testing_std=[]
    
    for ll in model_var:
        
        if game=='cartpole':
            envl = gym.make('CartPoleExtraLong-v0').unwrapped
            envl.length = ll
        elif game=='acrobot':
            envl = gym.make('AcrobotExtraLong-v0').unwrapped
            envl.LINK_LENGTH_1 = ll
        _=envl.seed(rs)
        envl.action_space.np_random.seed(rs)
    
        testing_all.append([])
        testing_avg.append([])
        testing_std.append([])
        print('*** Testing ***')
        
        # for all testing loops
        for k in range(test_loops):
            avg = 0
            std2 = 0
            
            # for nepochs epochs (episodes)
            for i in range(nepochs):
                
                score = 0
                
                # reset/observe current state
                state = envl.reset()
                state = torch.FloatTensor(state).to(device)
                
                # repeat until failure and up to time_steps
                for t in range(time_steps):
                    
                    # pick next action
                    policy = actor(state)
                    action = policy.sample()
                    
                    # observe new state
                    new_state, reward, done, info = envl.step(action.cpu().numpy())
                    new_state = torch.FloatTensor(new_state).to(device) 
                    score += reward
                    
                    # New State
                    state=new_state
                    
                    if done:
                        break
                
                # compute average over time_steps repeats 
                testing_all[-1].append(score)
                avg = avg + 1/(i+1) * (score-avg) # = avg * i/(i+1) + (t+1)/(i+1) 
                if i==0:
                    std2 = (score - avg)**2 
                else:
                    std2 = (i-1)/i * std2 + 1/(i+1) * (score - avg)**2             
                
            # Compute Average number of timesteps
            testing_avg[-1].append(avg)
            testing_std[-1].append(np.sqrt(std2))
            print(f'{(k+1)*nepochs}: Testing Average: {avg:.2f} +- {np.sqrt(std2):.2f}')
    
    #%% Plot Training Curve
    
    # colors = ['r','g','m','y','k','c','pink']
    colors = mcolors.TABLEAU_COLORS
    colors = list(colors.keys())
    
    fig = plt.figure(facecolor='white')
    
    plt.title(name)
    
    # x=np.arange(len(training_all))+1
    # y=np.array(training_all)
    # plt.plot(x,y,color='b',alpha=0.05)
    
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
    plt.plot(x,y, label='Training',color='k',linewidth=2)
    plt.fill(xfill,yfill,
              alpha=.1, fc='k', ec='None')
    
    for ll in range(len(testing_all)):
    
        color = colors[ll%len(colors)]
        
        # x=len(training_all)+np.arange(len(testing_all[ll]))+1
        # y=np.array(testing_all[ll])
        # plt.plot(x,y,color=color,alpha=0.05)
        
        x=(np.arange(len(testing_avg[ll]))+1)*nepochs + len(training_all)
        x=np.insert(x, 0, len(training_all))
        y=np.array(testing_avg[ll])
        y=np.insert(y,0,testing_avg[ll][0])
        sigma = np.array(testing_std[ll])
        sigma = np.insert(sigma,0,sigma[0])
        xfill = np.concatenate([x, x[::-1]])
        yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
        yfill = np.maximum(yfill,min(testing_all[ll])*np.ones_like(yfill))
        yfill = np.minimum(yfill,max(testing_all[ll])*np.ones_like(yfill))
        plt.plot(x,y, label=f'(l={model_var[ll]})',color=color,linewidth=2)
        plt.fill(xfill,yfill,
                  alpha=.1, fc=color, ec='None')
    
    plt.xlabel('Number of episodes')
    plt.ylabel('Reward')
    
    plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
    plt.legend()

    plot_file = results_folder+'/'+name+'.png'
    fig.savefig(plot_file, format = 'png')  
    
    plt.show()
        
    #%% Save results to file 
        
    my_results = [training_all, testing_all]
           
    results_file = results_folder+'/'+name+'.pkl'
    with open(results_file, mode='wb') as file:
        pickle.dump(my_results, file) 

#%% Run as standalone program

if __name__ == "__main__":
    train(risk_beta=0)
    train(risk_beta=0.1)
    train(risk_beta=-0.1)
    
#%%

"""
    Policy Gradient Algorithms for Risk-Sensitive Reinforcement Learning
    Christos Mavridis, Erfaun Noorani, John Baras <{mavridis,enoorani,baras}@umd.edu>
    Department of Electrical and Computer Engineering 
    University of Maryland
"""