#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import copy

#
# random utilties
#
def find_modules(stack, mod_name):
    matchidx = [i for i, m in enumerate(stack.modules) if m.name==mod_name]
 
    return matchidx

def save_model(stack, file_path):
    
    # truncate data to save disk space
    stack2=copy.deepcopy(stack)
    for i in range(1,len(stack2.data)):
        del stack2.data[i][:]
        
    directory = os.path.dirname(file_path)
    
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)       
        
    # Store data (serialize)
    with open(file_path, 'wb') as handle:
        pickle.dump(stack2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved model to {0}".format(file_path))

def load_model(file_path):
    try:
        # Load data (deserialize)
        with open(file_path, 'rb') as handle:
            stack = pickle.load(handle)
        return stack
    except:
        print("error loading {0}".format(file_path))
                   
    return stack

    

#
# PLOTTING FUNCTIONS
#
def plot_spectrogram(m,idx=None,size=(12,4)):
    #Moved from pylab to pyplot module in all do_plot functions, changed plots 
    #to be individual large figures, added other small details -njs June 16, 2017
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    if out1['stim'].ndim==3:
        plt.imshow(out1['stim'][:,m.parent_stack.plot_stimidx,:], aspect='auto', origin='lower')
    else:
        s=out1['stim'][m.parent_stack.plot_stimidx,:]
        r=out1['resp'][m.parent_stack.plot_stimidx,:]
        pred, =plt.plot(s,label='Predicted')
        resp, =plt.plot(r,'r',label='Response')
        plt.legend(handles=[pred,resp])
            
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def pred_act_scatter(m,idx=None,size=(12,4)):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1['stim'][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
    plt.plot(s,r,'ko')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def pred_act_psth(m,size=(12,4),idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1['stim'][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
    pred, =plt.plot(s,label='Predicted')
    act, =plt.plot(r,'r',label='Actual')
    plt.legend(handles=[pred,act])
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def pre_post_psth(m,size=(12,4),idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    in1=m.d_in[m.parent_stack.plot_dataidx]
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s1=in1['stim'][m.parent_stack.plot_stimidx,:]
    s2=out1['stim'][m.parent_stack.plot_stimidx,:]
    pre, =plt.plot(s1,label='Pre-nonlinearity')
    post, =plt.plot(s2,'r',label='Post-nonlinearity')
    plt.legend(handles=[pre,post])
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def plot_strf(m,idx=None,size=(12,4)):    
    if idx:
        plt.figure(num=idx,figsize=size)
    h=m.coefs
    plt.imshow(h, aspect='auto', origin='lower',cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.title(m.name)
