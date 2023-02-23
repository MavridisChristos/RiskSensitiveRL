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
#plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Load Results & Plot Parameters

title = 'var-param-ac'
game = 'cartpole'

names = ['LA01BL-BETA00-NN016/LR0100-Ao00-Ai50.pkl',
         'LA01-BETA01-NN016/LR0030-Ao00-Ai50.pkl',
         'LA01-BETA50-NN016/LR0030-Ao00-Ai50.pkl']

legends = ['Risk-Neutral',
           r'Risk-Sensitive $\beta=0.01$',
           r'Risk-Sensitive $\beta=-0.01$']

legends = ['Risk-Neutral',
           r'Risk-Sensitive $\beta=0.01$',
           r'Risk-Sensitive $\beta=-0.01$']

model_var = [round(0.2*(i+1),1) for i in range(10)]

p = 0.1

# Parameters
font_size = 34
line_width=7.0
marker_size=12.0
# fill_style = 'full' # 'full', 'none'
outercolors = ['k','b','r','g','m','y','c','pink','orange','brown',]
# colors = ['tab:blue',
#           'tab:orange',
#           'tab:green',
#           'tab:red',
#           'k',
#           'tab:purple',
#           'tab:brown',
#           'tab:pink',
#           'tab:gray',
#           'tab:olive',
#           'tab:cyan']
colors = mcolors.TABLEAU_COLORS
colors = list(colors.keys())
# colors.insert(4,'k')

colormaps = [plt.get_cmap('Blues'),plt.get_cmap('Greens'),plt.get_cmap('Reds')]


# plot folder
# folder = './results/'+game
folder = '../'
dfolder = folder+'/plots'
os.makedirs(dfolder, exist_ok=True)

#%% Plot

# Create new Figure
fig,ax = plt.subplots(figsize=(12, 8.5))

# plt.title(title,fontsize=font_size)

# Label axes
ax.set_ylim(0,210)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xlabel('Pole length', fontsize = font_size)
ax.set_ylabel('Reward', fontsize = font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size-4)

for n, name in enumerate(names):
    
    color = colormaps[n](0.9)
    
    file = folder+'/'+name
    with open(file, mode='rb') as file:
        training_all,testing_all = pickle.load(file)
        
        testing_avg = []
        testing_std = []
        testing_var = []
        testing_cvar = []
        for ll in range(len(testing_all)):
            
            # color = colors[ll]
            buffer = np.array(testing_all[ll])
            avg=buffer.mean()
            std=buffer.std()
            var=np.quantile(buffer,p)
            cvar=buffer[buffer<=var].mean()
            
            testing_avg.append(avg)
            testing_std.append(std)
            testing_var.append(var)
            testing_cvar.append(cvar)
            
            
        x=model_var
        y=testing_avg
        plt.plot(x,y,color=color,label=legends[n],marker='D',linewidth=line_width,markersize=marker_size,alpha=0.8)    

        x=np.concatenate([x, x[::-1]])
        y=np.concatenate([testing_cvar, testing_avg[::-1]])
        plt.fill(x,y,alpha=.1, fc=color, ec='None')
        
        # x=model_var
        # y=testing_cvar
        # plt.plot(x,y,color=color,marker='X',linewidth=line_width,markersize=marker_size,alpha=0.8)    

# Show Legend
plt.grid(color='gray', linestyle='-', linewidth=2, alpha = 0.2)
plt.legend(prop={'size': font_size})
plt.tight_layout()
plt.show()

# save figure
if True:
    fig.savefig(dfolder+'/'+title+'.png', format = 'png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
