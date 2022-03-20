#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Q-learning with Gym OpenAI
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

import cartpole

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Hyper-parameters 

# Low D or High D
lowD = True
# lowD = False

# Simulation Epochs
train_loops = 7 # hundreds of training epochs
test_loops = 3 # hundreds of testing epochs
nepochs = 100
time_steps = 500 # time horizon for successful epoch
update_once = True # update Q values during the first 100 epochs of testing loops

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
np.random.seed(rs)
random.seed(rs) 

#%% Q-learning Parameters 

if lowD:
    nnc=10
    nc = [nnc, nnc] # number of clusters # [theta, theta_dot]
    box = [0.15, 2.5] # box to discretize for initial conditions
else:
    nnc=5
    nc = [nnc, nnc, nnc, nnc] # number of clusters # [x, x_dot, theta, theta_dot]
    box = [1.0, 4.0, 1.0, 4.0] # box to discretize for initial conditions
gamma_Q = 0.9 # RL discount
epsilon = 0.2 # for epsilon-Greedy policy 
epercent = 0.8 # epsilon = epercent * epsilon
aa_init=0.1 # 0.9
aa_step = 0.3 # 0.9

#%% Environment Initialization and Random Seeds

env = cartpole.CartPoleEnv()
env_seed=env.seed(rs)
env.action_space.np_random.seed(rs)

#%% RL Model Initializtion
    
# Initialize clusters (may not be needed for ODA)
cluster_list=[]
for i in range(len(nc)):
    cluster_list.append(np.linspace(-box[i],box[i],nc[i])) 

cluster_size = np.prod(nc)

# Clusters: list of np.arrays
# clusters = np.zeros((cluster_size,len(nc)))
clusters = []
for i in range(cluster_size):
    tmp = []
    for j in range(len(nc)):
        tmp.append(0.0)
    clusters.append(np.array(tmp))

n=0
if lowD:
    for i in range(nc[0]):
        for j in range(nc[1]):
            clusters[n][0] = cluster_list[0][i]
            clusters[n][1] = cluster_list[1][j]
            n+=1
else:
    for i in range(nc[0]):
        for j in range(nc[1]):
            for k in range(nc[2]):
                for l in range(nc[3]):
                    clusters[n][0] = cluster_list[0][i]
                    clusters[n][1] = cluster_list[1][j]
                    clusters[n][2] = cluster_list[2][k]
                    clusters[n][3] = cluster_list[3][l]
                    n+=1

cluster_labels = [0 for i in range(cluster_size)]

Q = list(np.zeros(cluster_size*env.action_space.n).reshape((-1,env.action_space.n)))

def alpha(t):
    return 1/(1+aa_init+t*aa_step)
    # return 1.0/aa 

# find state representative in the set of codevectors
def q_state(state):
    # cluster_arr = np.array(clusters)
    # eucl_dist = np.sum((s-cluster_arr)**2,axis=1)
    eucl_dist = np.sum((state-clusters)**2,axis=1)
    qs = np.argmin( eucl_dist )
    return np.int_(qs)

def eGreedy(state, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore action space
    else:
        return np.argmax(Q[q_state(state)]) # Exploit learned values

def update_Q(t, state, action, new_state, cost):
    
    global Q
    
    Qold = Q[q_state(state)][action]
    Qnew = np.max(Q[q_state(new_state)]) 
    # Qsarsa = Q[tuple(q_state(new_state))][eGreedy(new_state,epsilon)]
    Q[q_state(state)][action] = Qold + \
                alpha(t) * ( cost + gamma_Q *Qnew - Qold )

def update_clusters(t, state, action, new_state, cost):
    
    global clusters
    global cluster_size
    global Q
    
    return

if False:
    for i in range(len(Q)):
        Q[i]=[0,0]
    epsilon=0.4
    time_steps=500

#%% Plot Initial State Space

def vis_V(q_state):
    
    three_D = False
    
    V = np.max(Q,axis=1) 

    # defining x, y, z co-ordinates 
    mesh_points = 101
    box2 = box[0]
    box3 = box[1]
    s2 = np.linspace(-box2, box2, mesh_points)
    s3 = np.linspace(-box3, box3, mesh_points)
    S2, S3 = np.meshgrid(s2, s3)
    
    # Plot the surface.
    V_plot = np.zeros((mesh_points,mesh_points))
    for i in range(mesh_points):
        for j in range(mesh_points):
            qs = q_state(np.array([s2[i],s3[j]])) 
            V_plot[i,j] = V[qs]
      
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
        x_label_list = [-box[0], 0, box[0]]
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(x_label_list)
        y_label_list = [-box[1], 0, box[1]]
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(y_label_list)
        fig.colorbar(img,shrink=0.3, aspect=5, pad=0.01)
    
if lowD and plot_figures:
    vis_V(q_state=q_state)
    plt.show()
    
#%% Training Loop

training_avg=[]
# for all training loops
for k in range(train_loops):
    avg = 0
    # for nepochs epochs
    for i in range(nepochs):
        
        # reset/observe current state
        state = env.reset()
        if lowD:
            state = state[2:]
        # repeat until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            action = eGreedy(state,epsilon)
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if lowD:
                new_state = new_state[2:]
            
            # Update RL model
            if (t+1)%((k+1)*5)==0: 
                update_clusters(t, state, action, new_state, cost)
            update_Q(t, state, action, new_state, cost)
            
            state=new_state
            if terminate:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        # compute average over 100 repeats    
        avg = avg + 1/(i+1) * (t+1-avg) # = avg * i/(i+1) + (t+1)/(i+1)              
    
    # Update epsilon    
    epsilon = epercent*epsilon
    
    # Compute Average number of timesteps
    training_avg.append(avg)
    print(f'{k+1}-th hundred (e={round(epsilon,2)}) : Average timesteps: {avg}. K = {len(Q)}')
    
    # Visualize Value Function
    if lowD and plot_figures:
        vis_V(q_state=q_state)
        plt.show()

#%% Testing Loop

if False:
    time_steps=1000
    test_loops=9
    
epsilon_test=1e-12
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
            action = eGreedy(state,epsilon_test)
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if lowD:
                new_state = new_state[2:]
            
            # Update RL model
            if update_once:
                update_Q(t, state, action, new_state, cost)
            
            state=new_state
            if terminate:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        
        # compute average over 100 repeats    
        avg = avg + 1/(i+1) * (t+1-avg) # = avg * i/(i+1) + (t+1)/(i+1)   
        
    update_once=False
    testing_avg.append(avg)
    print(f'Average Number of timesteps: {avg}')
    
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
    
    epsilon=1e-6
    avg = 0
    
    for i in range(1):
        
        state = env.reset()
        if lowD:
            state = state[2:]
            
        for t in range(time_steps):
            
            env.render()
        
            time.sleep(0.01)
                
            # pick next action
            action = eGreedy(state,epsilon_test)
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if lowD:
                new_state = new_state[2:]
            
            # Update RL model
            if update_once:
                update_Q(t, state, action, new_state, cost)
            
            state=new_state
            if terminate:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        avg = avg + 1/(i+1) * (t+1-avg)
        
    print(f'Average Number of timesteps: {avg}')
    env.close()