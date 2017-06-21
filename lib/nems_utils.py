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


def quick_plot_save(stack, mode=None):
    """Copy of quick_plot from nems_modules.stack for easy save or embed.
    
    mode options:
    -------------
        "json" -- .json
        "html" -- .html
        "png" -- .png
        default -- .png
        
    returns:
    --------
    filename : string
        Path to saved file, currently of the form:
        "/auto/data/code/nems_saved_models/batch{#}/{cell}_{modelname}.type"
    
    """
    batch = stack.meta['batch']
    cellid = stack.meta['cellid']
    modelname = stack.meta['modelname']
    
    fig = plt.figure(figsize=(8,9))
    for idx,m in enumerate(stack.modules):
        # skip first module
        if idx>0:
            plt.subplot(len(stack.modules)-1,1,idx)
            m.do_plot(m)
    
    if mode is not None:
        mode = mode.lower()
    if mode is None:
        filename = (
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.png"
            .format(batch,cellid,modelname)
            )
        fig.savefig(filename)
    elif mode == "png":
        filename = (
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.png"
            .format(batch,cellid,modelname)
            )
        fig.savefig(filename)
    elif mode == "json":
        filename = (
                "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.JSON"
                .format(batch,cellid,modelname)
                )
        mpld3.save_json(fig, filename)
    elif mode == "html":
        filename = (
                "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.html"
                .format(batch,cellid,modelname)
                )
        mpld3.save_html(fig, filename)
    else:
        filename = (
                "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.png"
                .format(batch,cellid,modelname)
                )
        fig.savefig(filename)
    # TODO: more file format options?
    # TODO: keep png as default, or something else more appropriate?
    plt.close(fig)
    return filename

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
        plt.imshow(out1['stim'][:,m.parent_stack.plot_stimidx,:], aspect='auto', origin='lower', interpolation='none')
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
    mmax=np.max(np.abs(h.reshape(-1)))
    plt.imshow(h, aspect='auto', origin='lower',cmap=plt.get_cmap('jet'), interpolation='none')
    plt.clim(-mmax,mmax)
    plt.colorbar()
    plt.title(m.name)