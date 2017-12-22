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

def plot_weights_64D(h, cellids,vmin, vmax,cbar=True):
    
    '''
    given a weight vector, h, plot the weights on the appropriate electrode channel
    mapped based on the cellids supplied. Weight vector must be sorted the same as
    cellids. Channels without weights will be ploted at empty dots
    '''
     # Make a vector for each column of electrodes
    
    # left column + right column are identical
    lr_col = np.arange(0,21*0.25,0.25)  # 25 micron vertical spacing
    left_ch_nums = np.arange(3,64,3)
    right_ch_nums = np.arange(4,65,3)
    center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)
    center_col = np.arange(-0.25,20.25*.25,0.25)-0.125
    ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
    sort_inds = np.argsort(ch_nums)
    
    
    
    l_col = np.vstack((np.ones(21)*-0.2,lr_col))
    r_col = np.vstack((np.ones(21)*0.2,lr_col))
    c_col = np.vstack((np.zeros(22),center_col))
    locations = np.hstack((l_col,c_col,r_col))[:,sort_inds]
    #plt.figure()
    plt.scatter(locations[0,:],locations[1,:],facecolor='none',edgecolor='k',s=70)
    plt.axis('scaled')
    plt.xlim(-.5,.5)
    
    # Now, color appropriately
    electrodes = np.zeros(len(cellids))
    c_id = np.zeros(len(cellids))
    for i in range(0, len(cellids)):
        electrodes[i] = int(cellids[i][0][-4:-2])
    electrodes = np.unique(electrodes)-1
    
    # move units when there are >1 on same electrode
    for i, weight in enumerate(h):
        c_id[i] = int(cellids[i][0][-4:-2])-1
        tf =1
        while tf==1:
             if (int(cellids[i][0][-1])>1 and int(c_id[i]+1) <64):
                 c_id[i] = int(c_id[i]+1)
                 if sum(c_id[i] == electrodes)>0:
                     tf=1
                 else:
                     tf = 0
             elif (int(cellids[i][0][-1])>1 and int(c_id[i]+1) >= 64):
                 print('im using the 2nd option')
                 c_id[i] = int(c_id[i]-1)
                 if sum(c_id[i] == electrodes)>0:
                     tf=1
                 else:
                     tf = 0
             else:
                 tf = 0
    
    import matplotlib
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h)
    #mappable.set_cmap('jet')
    colors = mappable.to_rgba(h)
    plt.scatter(locations[:,(c_id.astype(int))][0,:],locations[:,c_id.astype(int)][1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=70,edgecolor='none')
    if cbar is True:
        plt.colorbar(mappable)
        
        
        
# plotting utils fro 128ch 4-shank depth

def plot_weights_128D(h, cellids):
    # get gemoetry from Luke's baphy function probe_128D
    
    channels = np.arange(0,128,1)
    x = loadmat('probe_128D/x_positions.mat')['x_128']
    y = loadmat('probe_128D/z_positions.mat')['z_128']
    
    locations=np.hstack((x,y))
    plt.scatter(locations[:,0],locations[:,1])
    plt.axis('scaled')

    
    
    
    
    