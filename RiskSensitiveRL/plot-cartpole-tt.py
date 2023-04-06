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

name='jair-tt'
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
# test_var = [0.0]
test_var = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

actual_model_var = 0.5
goal_max = 200
goal_min = 0

risk_betas = [0,0.001,-0.001,0.005,-0.005,0.01,-0.01]
risk_betas = [0,0.01,-0.01,0.05,-0.05,0.1,-0.1]
risk_betas = [-0.001]

learning_rates = [0.0003,0.0005, 0.0007,0.001]
learning_rates = [0.003,0.005, 0.007,0.01]
learning_rates = [0.0003]

b = 0
random_seeds = [b+0,b+1,b+2,b+3,b+4,b+5,b+6,b+7,b+8,b+9]
random_seeds = [b+0]

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

for risk_beta in risk_betas:
    
    for lr in learning_rates:
        
        traning_data = []
        testing_data = []
        wa_data = []
        wc_data = []
        
        no_file = False
        
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
                break
                
            with open(file, mode='rb') as file:
                training_all,testing_all, wa, wc = pickle.load(file)
            
            traning_data.append(training_all)
            testing_data.append(testing_all)
            wa_data.append(wa)
            wc_data.append(wc)
        
        if no_file:
            break

        fig,ax = plt.subplots(facecolor='white',figsize=(7,5),tight_layout = {'pad': 1})
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        colors = mcolors.TABLEAU_COLORS
        colors = list(colors.keys())
        pxmin = -150
        pxmax = 3250
        pymin = -20
        pymax = 220
        plt.xlim([-150, 3350])
        plt.ylim([-20, 220])
        
        # random seed average    
        training_all = []
        training_all_std = []
        for i in range(len(traning_data[0])):
            data = np.array([td[i] for td in traning_data])
            training_all.append(data.mean())
            training_all_std.append(data.std())
        
        # loops = int(len(training_all)/batch)
        
        # training_avg=[]
        # training_std=[]
        # j = 0
        # for k in range(loops):
        #     buffer_m = []
        #     buffer_s = []
        #     for i in range(batch):
        #         buffer_m.append(training_all[j])
        #         buffer_s.append(training_all_std[j])
        #         j+=1
        #     buffer_m = np.array(buffer_m)
        #     buffer_s = np.array(buffer_s)
        #     avg = buffer_m.mean()
        #     std = buffer_s.mean()
        #     training_avg.append(avg)
        #     training_std.append(std)

        training_avg=training_all[0::batch]
        training_std=training_all_std[0::batch]
        
        x=(np.arange(len(training_avg))+1)*batch
        x=np.insert(x, 0, 1)
        y=np.array(training_avg)
        y=np.insert(y,0,training_all[0])
        
        sigma = np.array(training_std)
        sigma = np.insert(sigma,0,sigma[0])
        xfill = np.concatenate([x, x[::-1]])
        yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
        yfill = np.maximum(yfill,goal_min*np.ones_like(yfill))
        yfill = np.minimum(yfill,goal_max*np.ones_like(yfill))
        
        if risk_beta>0:
            color = colors[4]
        elif risk_beta<0:
            color = colors[0]
        else:
            color = colors[7]
        plt.plot(x,y, label=r'$\beta=$'+f'{risk_beta}',color=color,linewidth=3,alpha=0.7, marker='D', markersize = 5)
        plt.fill(xfill,yfill,
                  alpha=.05, fc='k', ec='None')
        plt.fill_between([x[0],x[-1]],[goal_max+10,goal_max+10],[goal_min-15,goal_min-15],alpha=0.03,color='k')
        # plt.fill_between([pxmin,2000],[pymax,pymax],[pymin,pymin],alpha=0.03,color='k')
        
        # y=np.array(training_all_cvar[0::batch])
        # y=np.insert(y,0,training_all[0])
        # plt.plot(x,y,color='k',linewidth=2,alpha=0.7,linestyle=':')
        
        for ll in range(len(testing_data[0])):
            
            if model_var[ll] in test_var:
                # random seed average    
                testing_all = []
                testing_all_std = []
                for i in range(len(testing_data[0][ll])):
                    data = np.array([td[ll][i] for td in testing_data]) # sum across all seeds
                    testing_all.append(data.mean()) 
                    testing_all_std.append(data.std())
                data = np.array(sum([td[ll] for td in testing_data],[])) # sum across all seeds and episodes
                # data = np.array(testing_all)
                testing_all_var = np.quantile(data,p)
                testing_all_cvar = data[data<=testing_all_var].mean()
                testing_all_var9 = np.quantile(data,1-p)
                testing_all_cvar9 = data[data<=testing_all_var9].mean()
                
                # loops = int(len(testing_all)/batch)
                
                # testing_avg=[]
                # testing_std=[]
                # j = 0
                # for k in range(loops):
                #     buffer_m = []
                #     buffer_s = []
                #     for i in range(batch):
                #         buffer_m.append(testing_all[j])
                #         buffer_s.append(testing_all_std[j])
                #         j+=1
                #     buffer_m = np.array(buffer_m)
                #     buffer_s = np.array(buffer_s)
                #     avg = buffer_m.mean()
                #     std = buffer_s.mean()
                #     testing_avg.append(avg)
                #     testing_std.append(std)
                    
                testing_avg=testing_all[0::batch]
                testing_std=testing_all_std[0::batch]
                
                x=(np.arange(len(testing_avg))+1)*batch + len(training_all)
                x=np.insert(x, 0, len(training_all))
                y=np.array(testing_avg)
                y=np.insert(y,0,testing_all[0])
                
                sigma = np.array(testing_std)
                sigma = np.insert(sigma,0,sigma[0])
                xfill = np.concatenate([x, x[::-1]])
                yfill = np.concatenate([y - sigma, (y + sigma)[::-1]]) # 1.96
                yfill = np.maximum(yfill,goal_min*np.ones_like(yfill))
                yfill = np.minimum(yfill,goal_max*np.ones_like(yfill))
                
                tci = [9,8,6,5,3,2,1]
                tcolors = [colors[tcii] for tcii in tci]
                plt.plot(x,y, color=tcolors[ll],linewidth=3,alpha=0.7, marker='s', markersize = 5) #label=f'l={2*(actual_model_var + model_var[ll]):.1f}',
                plt.fill(xfill,yfill,
                          alpha=.05, fc=tcolors[ll], ec='None')
                
                if model_var[ll] == 0:
                # if True:
                    color = tcolors[ll]
                    xx = len(training_all) + len(testing_all)
                    yy = testing_all_cvar
                    plt.plot(xx,yy,label = f'CVaR$_{ {p} }={yy:.2f}$',color=color,linewidth=2,alpha=0.7,marker='.',markersize=5)
                    # plt.plot(xx,yy,color=color,linewidth=2,alpha=0.7,marker='_',markersize=75)
                    # plt.text(xx-0,yy-10,f'CVaR$_{ {p} }$',color=color, fontsize = 10, fontweight='bold')
                    plt.text(xx+100,yy-10,f'CVaR$_{ {p} }$',color=color,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold',
                              bbox=dict(boxstyle="round",
                                    fc=(1., 1., 1.),
                                    ec=(.1, .1, .1),alpha=0.5
                                    ),alpha=0.99)    
                    plt.plot(xx,yy,color=color,linewidth=2,alpha=0.7,marker='_',markersize=75)
                    
                    color = tcolors[ll]
                    yy = testing_all_cvar9
                    plt.plot(xx,yy,label = f'CVaR$_{ {1-p} }={yy:.2f}$',color=color,linewidth=2,alpha=0.7,marker='.',markersize=5)
                    # plt.plot(xx,yy,color=color,linewidth=2,alpha=0.7,marker='_',markersize=75)
                    # plt.text(xx-0,yy+5,f'CVaR$_{ {1-p} }$',color=color, fontsize = 10, fontweight='bold')
                    plt.text(xx+100,yy+9,f'CVaR$_{ {1-p} }$',color=color,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold',
                              bbox=dict(boxstyle="round",
                                    fc=(1., 1., 1.),
                                    ec=(.1, .1, .1),alpha=0.5
                                    ),alpha=0.99)    
                    plt.plot(xx,yy,color=color,linewidth=2,alpha=0.7,marker='_',markersize=75)
                    
        plt.fill_between([x[0],x[-1]],[goal_max+10,goal_max+10],[goal_min-15,goal_min-15],alpha=0.03,color='r')
        # plt.fill_between([2000,pxmax],[pymax,pymax],[pymin,pymin],alpha=0.03,color='r')
        
        tp = train_loops/(train_loops+test_loops)
        plt.text(tp-0.17,.05, 'Training Phase', horizontalalignment='center',verticalalignment='center', fontsize=12,#fontweight='bold',
                  transform = ax.transAxes,bbox=dict(boxstyle="round",
                        fc=(1., 1.0, 1.0),
                        ec=(0.1, .1, .1),alpha=0.1
                        ),alpha=0.7)    
        
        plt.text(tp+0.06,.05, 'Testing Phase', horizontalalignment='center',verticalalignment='center', fontsize=12,#fontweight='bold',
                  transform = ax.transAxes,bbox=dict(boxstyle="round",
                        fc=(1., 1.0, 1.0),
                        ec=(0.1, .1, .1),alpha=0.1
                        ),alpha=0.7)
        
        plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
        # plt.legend(loc='lower left',prop={'size': 14},framealpha=0.51, borderpad=1) #loc='upper left',
        plt.legend(prop={'size': 14},framealpha=0.51, borderpad=1) #loc='upper left',

        plt.xlabel('# Episodes', fontsize = 20)
        plt.ylabel('Reward', fontsize = 20)
        
        plot_file = '../results/'+game+'/'+folder+'/'+name+'.png'
        fig.savefig(plot_file, format = 'png')  
        plt.show()
        plt.close()
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
