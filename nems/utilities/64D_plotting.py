#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:13:38 2017

@author: hellerc
"""

# Plotting utilities for 64D Masmanidis array
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_from_mat(h_filename, ids_filename, lv):
    h = loadmat(h_filename)
    for key in h.keys():
        ht = h[key]
    h = ht
    cellids = loadmat(ids_filename)
    
    for key in cellids.keys():
        cellidst = cellids[key]
    cellids = cellidst

    plot_weights_64D(h[:,lv].squeeze(), cellids[0])

def plot_weights_64D(h, cellids):
    
    '''
    given a weight vector, h, plot the weights on the appropriate electrode channel
    mapped based on the cellids supplied. Weight vector must be sorted the same as
    cellids. Channels without weights will be ploted at empty dots
    '''
    
     # Make a vector for each column of electrodes
    
    # left column + right column are identical
    lr_col = np.arange(0,21*0.25,0.25)  # 25 micron vertical spacing
    left_ch_nums = np.append(np.arange(3,64,3),0)
    right_ch_nums = np.append(np.arange(4,65,3),0)
    center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)
    center_col = np.arange(-0.25,20.25*.25,0.25)-0.125
    ch_nums = np.vstack((left_ch_nums, center_ch_nums, right_ch_nums))
    
    plt.figure()
    plt.plot(np.zeros(22),center_col,'o',color='k',mfc='none')
    plt.plot(np.zeros(21)+0.2,lr_col,'o',color='k',mfc='none')
    plt.plot(np.zeros(21)-0.2,lr_col, 'o',color='k',mfc='none')
    plt.axis('scaled')
    plt.xlim(-.5,.5)
    
    # Now, color appropriately
    electrodes = np.zeros(len(cellids))
    for i in range(0, len(cellids)):
        electrodes[i] = int(cellids[i][0][-4:-2])
    electrodes = np.unique(electrodes)
  
    c = plt.get_cmap('jet')
    
    # move units when there are >1 on same electrode
    for i, weight in enumerate(h):
        c_id = int(cellids[i][0][-4:-2])
        tf =1
        while tf==1:
             if int(cellids[i][0][-1])>1:
                 c_id = c_id+1
                 if sum(c_id == electrodes)>0:
                     tf=1
                 else:
                     tf = 0
             else:
                 tf = 0
            
       
        location = np.argwhere(ch_nums == c_id)
        if location[0][0] == 0:
            plt.plot(-0.2,lr_col[location[0][1]],'.', color=c(h[i]),markersize=8)
            
        elif location[0][0]==1:
            plt.plot(0,center_col[location[0][1]],'.', color=c(h[i]),markersize=8)
            
        elif location[0][0]==2:
            plt.plot(0.2,lr_col[location[0][1]],'.', color=c(h[i]),markersize=8)
    
    
    