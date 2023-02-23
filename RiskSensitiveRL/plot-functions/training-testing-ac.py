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

title = 'training-testing-ac'
game = 'cartpole'

names = ['LA01BL-BETA00-NN016/LR0100-Ao00-Ai50.pkl',
         'LA01-BETA01-NN016/LR0030-Ao00-Ai50.pkl',
         'LA01-BETA50-NN016/LR0030-Ao00-Ai50.pkl']

legends = ['Risk-Neutral',
           r'Risk-Sensitive $\beta=0.01$',
           r'Risk-Sensitive $\beta=-0.01$']

batch = 100
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
fig,ax = plt.subplots(figsize=(14, 8.5))

# plt.title(title,fontsize=font_size)
model_var = [round(0.2*(i+1),1) for i in range(10)]

# Label axes
# ax1.set_ylim(0-30,2700)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xlabel('# Episodes', fontsize = font_size)
ax.set_ylabel('Reward', fontsize = font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size-4)

# ax.plot(x, y, label='None', 
#          color='k', marker='s',linestyle='solid', 
#          linewidth=line_width, markersize=marker_size, fillstyle=fill_style)

for n, name in enumerate(names):
    
    colors = colormaps[n]
    
    file = folder+'/'+name
    with open(file, mode='rb') as file:
        training_all,testing_all = pickle.load(file)
        
        color = colors(0.9)
        loops = int(len(training_all)/batch)
        
        training_avg=[]
        training_std=[]
        j = 0
        for k in range(loops):
            buffer = []
            for i in range(batch):
                buffer.append(training_all[j])
                j+=1
            buffer = np.array(buffer)
            avg = buffer.mean()
            std = buffer.std()
            var = np.quantile(buffer,p)
            cvar = buffer[buffer<=var].mean()
            training_avg.append(avg)
            training_std.append(std)
        
        
        
        # x=np.arange(len(training_all))+1
        # y=np.array(training_all)
        # ax.plot(x,y,color=colors[i],
        #         alpha=0.05,linewidth=1)
        
        x=(np.arange(len(training_avg))+1)*batch
        x=np.insert(x, 0, 1)
        y=np.array(training_avg)
        y=np.insert(y,0,training_all[0])
        
        sigma = np.array(training_std)
        sigma = np.insert(sigma,0,sigma[0])
        xfill = np.concatenate([x, x[::-1]])
        yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
        yfill = np.maximum(yfill,min(training_all)*np.ones_like(yfill))
        yfill = np.minimum(yfill,max(training_all)*np.ones_like(yfill))
        
        ax.plot(x,y,label=legends[n],color=color,
                alpha=0.8,marker='D',markersize=marker_size,linewidth=line_width)
        plt.fill(xfill,yfill,
                  alpha=.08, fc=color, ec='None')
        
        # for ll in range(len(testing_all)):
        for ll in [0,1,2,3,4]:
            
            color = colors(1-(0.1+(ll+1)*0.02))
            loops = int(len(testing_all[ll])/batch)
            testing_avg=[]
            testing_std=[]
            j = 0
            for k in range(loops):
                buffer = []
                for i in range(batch):
                    buffer.append(testing_all[ll][j])
                    j+=1
                buffer = np.array(buffer)
                avg = buffer.mean()
                std = buffer.std()
                var = np.quantile(buffer,p)
                cvar = buffer[buffer<=var].mean()
                testing_avg.append(avg)
                testing_std.append(std)
                
            # x=len(training_all)+np.arange(len(testing_all[ll]))+1
            # y=np.array(testing_all[ll])
            # plt.plot(x,y,color=color,alpha=0.05)
            
            x=(np.arange(len(testing_avg))+1)*batch + len(training_all)
            x=np.insert(x, 0, len(training_all))
            y=np.array(testing_avg)
            y=np.insert(y,0,testing_avg[0])
            sigma = np.array(testing_std)
            sigma = np.insert(sigma,0,sigma[0])
            xfill = np.concatenate([x, x[::-1]])
            yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
            yfill = np.maximum(yfill,min(testing_all[ll])*np.ones_like(yfill))
            yfill = np.minimum(yfill,max(testing_all[ll])*np.ones_like(yfill))
            # plt.plot(x,y, label=f'(l={model_var[ll]})',color=color,linewidth=2)
            plt.plot(x,y,color=color,linewidth=2,alpha=0.8)
            plt.fill(xfill,yfill,
                      alpha=.01, fc=color, ec='None')
        
        

# Show Legend
plt.grid(color='gray', linestyle='-', linewidth=2, alpha = 0.2)
plt.legend(prop={'size': font_size},loc='lower right')
plt.tight_layout()
plt.show()

# save figure
if True:
    fig.savefig(dfolder+'/'+title+'.png', format = 'png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
