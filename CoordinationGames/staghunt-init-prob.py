#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Stag-Hunt Game model
    Erfaun Noorani, Mavridis Christos, John S. Baras 
    Department of Electrical and Computer Engineering 
    University of Maryland
    <enoorani,mavridis,baras>@umd.edu
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
colors = mcolors.TABLEAU_COLORS
colors = list(colors.keys())

#%%

A = 5
B = 4
C = 0
D = 2

def rs(A,gamma=0.01):
    return np.exp(gamma*A)

Pmixed = []
PmixedRS = []

gammas = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
mgammas = [-g for g in reversed(gammas)]
allgammas = mgammas + gammas
for gamma in allgammas:

    Pmixed.append( (D-C)/(A-B+D-C) )
    PmixedRS.append( (rs(D,gamma)-rs(C,gamma))/(rs(A,gamma)-rs(B,gamma)+rs(D,gamma)-rs(C,gamma)) )

fig,ax = plt.subplots(figsize=(8,5),tight_layout = {'pad': 1})

plt.plot(allgammas,Pmixed,color=colors[0],label='Risk Neutral PG',linewidth=9.0,marker='o',markersize=8.5,alpha=0.5)
plt.plot(allgammas,PmixedRS,color=colors[1],label='Risk Sensitive PG',linewidth=10.5,marker='D',markersize=10.0,alpha=0.5)

ax.set_ylabel(r'$\alpha^*$ for (A,a) NE', fontsize = 26)
ax.set_xlabel(r'Risk Parameter $\gamma$', fontsize = 26)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=8)
        
plt.legend(prop={'size': 24})
plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
plt.show()

fig.savefig('./staghunt-init-prob.png', format = 'png')

    
    