#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Import Modules

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import pylab as pl

plt.ion()
plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Load Results & Plot Parameters

name='jair-bb'
game = 'cartpole'
look_ahead = 1
baseline = False
risk_objective = 'BETA' 
train_loops = 200
test_loops = 100
nepochs = 10
time_steps = 200 
nn_actor = [16]
nn_critic = [16]
a_outer = 0.0
a_inner = 0.0 
cut_lr=0
model_var = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
test_var = [0.0]

actual_model_var = 0.5
goal_max = 200
goal_min = 0

risk_betas = [0,0.001,-0.001,0.005,-0.005,0.01,-0.01]
# risk_betas = [0,0.01,-0.01,0.05,-0.05,0.1,-0.1]
# risk_betas = [-0.001]

learning_rates = [0.0003,0.0005, 0.0007,0.001]
learning_rates = [0.003,0.005, 0.007,0.01]
learning_rates = [0.0003]

b = 0
random_seeds = [b+0,b+1,b+2,b+3,b+4,b+5,b+6,b+7,b+8,b+9]
# random_seeds = [b+0]

batch = 25
p = 0.1

#%% Filenames

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
    elif d==5:
        text+=f'{int(x):05}' 
    return text

def b2t(b,text='bl'):
    if b:
        return text
    else:
        return ''
    
def filename(game,
            look_ahead,
            baseline,
            risk_beta,
            risk_objective, 
            nn_actor,
            nn_critic,
            lr,
            a_outer,
            a_inner, 
            cut_lr,
            rs):
    
    folder_name = 'LA'+n2t(look_ahead)+b2t(baseline,'BL')+'-'+risk_objective+n2t(risk_beta*1000,4)+'-NN'+n2t(np.sum(nn_actor),3)+n2t(np.sum(nn_critic),3)+'/'+ \
            'LR'+b2t(cut_lr,'CUT')+n2t(lr*100000,5)+'-Ao'+n2t(a_outer*100,2)+'-Ai'+n2t(a_inner*100,2)
    # folder_name = 'LA'+n2t(look_ahead)+b2t(baseline,'BL')+'-'+risk_objective+n2t(risk_beta*1000,4)+'-NN'+n2t(np.sum(nn_actor),3)+n2t(np.sum(nn_critic),3)+'/'+ \
    #         'LR'+n2t(lr*100000,5)+'-CUT'+n2t(cut_lr,1)+'-Ao'+n2t(a_outer*100,2)+'-Ai'+n2t(a_inner*100,2)
    name=folder_name+'/'+'RS'+n2t(rs,2)
    
    return name, folder_name

#%% Plot

bmeans = []
bvar = []
bcvar = []
bcvar9 = []

for risk_beta in risk_betas:
    
    max_mean = 0
    max_var = 0
    max_cvar = 0
    max_cvar9 = 0
    
    for lr in learning_rates:
        
        no_file = False
        
        testing_data = []
        
        for rs in random_seeds:
            
            file, folder = filename(game,
                        look_ahead,
                        baseline,
                        risk_beta,
                        risk_objective, 
                        nn_actor,
                        nn_critic,
                        lr,
                        a_outer,
                        a_inner, 
                        cut_lr,
                        rs)
            
            file = '../results/'+game+'/'+file+'.pkl'
            if not os.path.isfile(file):
                no_file = True
                print('No File '+file)
                break
                
            with open(file, mode='rb') as file:
                training_all,testing_all, wa, wc = pickle.load(file)
            
            testing_data.append(testing_all)
            
        if no_file:
            break

        for ll in range(len(testing_data[0])):
            
            if model_var[ll] in test_var:
                
                data = np.array(sum([td[ll] for td in testing_data],[])) # sum across all seeds and episodes
                # data = np.array(testing_all)
                testing_all_mean = data.mean()
                testing_all_var = np.quantile(data,p)
                testing_all_cvar = data[data<=testing_all_var].mean()
                testing_all_var9 = np.quantile(data,1-p)
                testing_all_cvar9 = data[data<=testing_all_var9].mean()

        max_mean = max(max_mean, testing_all_mean)
        max_var = min(max_var, testing_all_var)
        max_cvar = max(max_cvar, testing_all_cvar)
        max_cvar9 = max(max_cvar9, testing_all_cvar9)

    bmeans.append(max_mean)
    bvar.append(max_var)
    bcvar.append(max_cvar)
    bcvar9.append(max_cvar9)


fig,ax = plt.subplots(facecolor='white',figsize=(6,4),tight_layout = {'pad': 1})
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=8)
colors = mcolors.TABLEAU_COLORS
colors = list(colors.keys())
markers = ['s','D','o','X','P']
# plt.xlim([-50, 3250])
plt.ylim([-5, 210])



x = risk_betas
ax.set_xticks([x[0],x[-1],x[-2]])
ax.set_yticks([50,100,150,200])
bnames = ['Average',f'CVaR$_{ {p} }$',f'CVaR$_{ {1-p} }$']
bmarkers = [12,8,4]
bcolors=[colors[bc] for bc in [0,8,3]]
blines = ['dashdot','dashed','dotted']
# for idy,y in enumerate([bmeans,bcvar,bcvar9]):

#     for idx, xx in enumerate(x):
        
#         color = bcolors[idy%len(bcolors)]
#         marker = markers[idy%len(markers)]
#         # plt.bar(xx,y[idx],width=0.001,alpha=0.5,color=color) # label=r'$\beta$'+f'={xx}',
        
#         if idx==0:
#             plt.plot(xx,y[idx],label=f'{bnames[idy]}',color=color,alpha=0.7, 
#                       marker=marker, markersize = bmarkers[idy]) 
#         else:
#             plt.plot(xx,y[idx], color=color,alpha=0.7, 
#                       marker=marker, markersize = bmarkers[idy]) 

ixx = np.argsort(risk_betas)  
x = [risk_betas[iixx] for iixx in ixx]
y = [[yy[iixx] for iixx in ixx] for yy in [bmeans,bcvar,bcvar9]]
for idy,yy in enumerate(y):
    color = bcolors[idy%len(bcolors)]
    marker = markers[idy%len(markers)]
    plt.plot(x,yy,label=f'{bnames[idy]}',color=color,alpha=0.7, 
                  marker=marker, markersize = bmarkers[idy],linestyle=blines[idy]) 
plt.fill_between(x,y[0],y[1],alpha=0.1,color=bcolors[0])
            
plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
plt.legend(prop={'size': 14},framealpha=0.51, borderpad=1)

plt.xlabel(r'$\beta$', fontsize = 22)
plt.ylabel('Reward', fontsize = 22)

plot_file = '../results/'+game+'/'+name+'-'+n2t(look_ahead,1)+'.png'
fig.savefig(plot_file, format = 'png')  
plt.show()
plt.close()
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
