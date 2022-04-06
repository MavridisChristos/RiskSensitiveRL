#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Import Modules

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import pylab as pl

plt.ion()
#plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Load Results & Plot Parameters

title = 'cartpole-std-3'

names = ['nonex.pkl',
         'b001.pkl',
         'b001m.pkl']

legends = ['Risk-Neutral',
           r'$\beta=0.01$',
           r'$\beta=-0.01$']

# Parameters
font_size = 22
line_width=7.0
marker_size=15.0
fill_style = 'full' # 'full', 'none'
colors = ['b','k','r','g','m','y','c','pink']

# plot folder
folder = './results/cartpole'
dfolder = './plots'
os.makedirs(dfolder, exist_ok=True)

#%% Figure

# Create new Figure
fig,ax = plt.subplots(figsize=(12, 8.5))

# plt.title(title,fontsize=font_size)

# Label axes
ax.set_ylim(0,30)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xlabel('# Episodes', fontsize = font_size)
ax.set_ylabel('Variance', fontsize = font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size-4)

#%% Plot

# ax.plot(x, y, label='None', 
#          color='k', marker='s',linestyle='solid', 
#          linewidth=line_width, markersize=marker_size, fillstyle=fill_style)

for n, name in enumerate(names):
    file = folder+'/'+name
    with open(file, mode='rb') as file:
        _,_,_,training_all,training_avg,training_std = pickle.load(file)
        
        training_avg=[]
        training_std=[]
        j = 0
        for k in range(3):
            avg = 0
            std2=0
            for i in range(160):
                score = training_all[j]
                j+=1
                avg = avg + 1/(i+1) * (score-avg) 
                if i==0:
                    std2 = (score - avg)**2 
                else:
                    std2 = (i-1)/i * std2 + 1/(i+1) * (score - avg)**2  
            training_avg.append(avg)
            training_std.append(np.sqrt(std2))
        
        # x=np.arange(len(training_all))+1
        # y=np.array(training_all)
        # ax.plot(x,y,color=colors[i],
        #         alpha=0.05,linewidth=1)
        
        x=(np.arange(len(training_std)))*500
        # x=np.insert(x, 0, 1)
        y=np.array(training_std)
        # y=np.insert(y,0,training_std[-1])
        
        xfill = np.concatenate([x, x[::-1]])
        yfill = np.concatenate([y - 100, (y)[::-1]]) # 1.96
        yfill = np.maximum(yfill,0*np.ones_like(yfill))
        # yfill = np.minimum(yfill,0*np.ones_like(yfill))
        
        ax.plot(x,y,label=legends[n],color=colors[n],
                alpha=0.8,linewidth=line_width)
        plt.fill(xfill,yfill,
                  alpha=.1, fc=colors[n], ec='None')

# Show Legend
plt.grid(color='gray', linestyle='-', linewidth=2, alpha = 0.2)
plt.legend(prop={'size': font_size+4},loc='upper right')
plt.show()

# save figure
if True:
    fig.savefig(dfolder+'/'+title+'.png', format = 'png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
