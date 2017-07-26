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

# set default figsize for pyplots (so we don't have to change each function)
FIGSIZE=(12,4)

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
    
    try:
    # Store data (serialize)
        with open(file_path, 'wb') as handle:
            pickle.dump(stack2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except FileExistsError:
        print("Removing existing model at: {0}".format(file_path))
        os.remove(file_path)
        with open(file_path, 'wb') as handle:
            pickle.dump(stack2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    #orig_umask = os.umask(0)
    #try:
    #    file=os.open(
    #            path=file_path,
    #            flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    #            mode=0o777,
    #            )
    #finally:
    #    os.umask(orig_umask)

    #with os.fdopen(file, 'wb') as handle:
    #    pickle.dump(stack2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved model to {0}".format(file_path))

def load_model(file_path):
    try:
        # Load data (deserialize)
        with open(file_path, 'rb') as handle:
            stack = pickle.load(handle)
        return stack
    except:
        # TODO: need to do something else here maybe? removed return stack
        #       at the end b/c it was being returned w/o assignment when
        #       open file failed.
        print("error loading {0}".format(file_path))
        return


#
# PLOTTING FUNCTIONS
#
#TODO: find some way to get the stimuli to resolve correctly for the pupil model stacks,
#since stack.data[1] will have ~2 stimuli, but it is subsequently reshaped to ~240 stimuli
# ---fixed, see except statement in plot_spectrogram --njs 6 July 2017

def plot_spectrogram(m,idx=None,size=FIGSIZE):
    #Moved from pylab to pyplot module in all do_plot functions, changed plots 
    #to be individual large figures, added other small details -njs June 16, 2017
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    reps=out1['repcount']
    ids=m.parent_stack.plot_stimidx
    r=reps.shape[0]
    lis=[]
    for i in range(0,r):
        lis.extend([i]*reps[i])
    new_id=lis[ids]
    if out1['stim'].ndim==3:
        try:
            plt.imshow(out1['stim'][:,m.parent_stack.plot_stimidx,:], aspect='auto', origin='lower', interpolation='none')
        except:
            plt.imshow(out1['stim'][:,new_id,:], aspect='auto', origin='lower', interpolation='none')
        cbar = plt.colorbar()
        #cbar.set_label('???')
        # TODO: colorbar is intensity of response? but how is it measured?
        plt.xlabel('Trial')
        plt.ylabel('Channel')
    else:
        s=out1['stim'][:,new_id]
        #r=out1['resp'][m.parent_stack.plot_stimidx,:]
        pred, =plt.plot(s,label='Average Model')
        #resp, =plt.plot(r,'r',label='Response')
        plt.legend(handles=[pred])
        plt.xlabel('Time Step')
        plt.ylabel('Firing rate (unitless)')
            
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def pred_act_scatter(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1['stim'][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
    plt.plot(s,r,'ko')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def pred_act_psth(m,size=FIGSIZE,idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1['stim'][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
    pred, =plt.plot(s,label='Predicted')
    act, =plt.plot(r,'r',label='Actual')
    plt.legend(handles=[pred,act])
    plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Trial')
    plt.ylabel('Firing rate (unitless)')


def pre_post_psth(m,size=FIGSIZE,idx=None):
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
    plt.xlabel('Trial')
    plt.ylabel('Firing rate (unitless)')

def plot_stim_psth(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=str(idx),figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    #c=out1['repcount'][m.parent_stack.plot_stimidx]
    h=out1['stim'][m.parent_stack.plot_stimidx].shape
    #scl=int(h[0]/c)
    s2=out1['stim'][m.parent_stack.plot_stimidx,:]
    resp, =plt.plot(s2,'r',label='Post-'+m.name)
    plt.legend(handles=[resp])
    plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx))
  
def plot_strf(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
    h=m.coefs
    mmax=np.max(np.abs(h.reshape(-1)))
    plt.imshow(h, aspect='auto', origin='lower',cmap=plt.get_cmap('jet'), interpolation='none')
    plt.clim(-mmax,mmax)
    cbar = plt.colorbar()
    #TODO: cbar.set_label('???')
    plt.title(m.name)
    plt.xlabel('Channel') #or kHz?
    # Is this correct? I think so...
    plt.ylabel('Latency') 
"""
Potentially useful trial plotting stuff...not currently in use, however --njs July 5 2017   
def plot_trials(m,idx=None,size=(12,4)):
    out1=m.d_out[m.parent_stack.plot_dataidx]
    u=0
    c=out1['repcount'][m.parent_stack.plot_stimidx]
    h=out1['stim'][m.parent_stack.plot_stimidx].shape
    scl=int(h[0]/c)
    tr=m.parent_stack.plot_trialidx
    
    #Could also rewrite this so all plots are in a single figure (i.e. each 
    #trial is a subplot, rather than its own figure)
    for i in range(tr[0],tr[1]):
        plt.figure(num=str(idx)+str(i),figsize=size)
        s=out1['stim'][m.parent_stack.plot_stimidx,u:(u+scl)]
        r=out1['resp'][m.parent_stack.plot_stimidx,u:(u+scl)]
        pred, =plt.plot(s,label='Predicted')
        resp, =plt.plot(r,'r',label='Response')
        plt.legend(handles=[pred,resp])
        plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx)+', trial #'+str(i))
        u=u+scl
        
def trial_prepost_psth(m,idx=None,size=(12,4)):
    in1=m.d_in[m.parent_stack.plot_dataidx]
    out1=m.d_out[m.parent_stack.plot_dataidx]
    u=0
    c=out1['repcount'][m.parent_stack.plot_stimidx]
    h=out1['stim'][m.parent_stack.plot_stimidx].shape
    scl=int(h[0]/c)
    tr=m.parent_stack.plot_trialidx
    
    for i in range(tr[0],tr[1]):
        plt.figure(num=str(idx)+str(i),figsize=size)
        s1=in1['stim'][m.parent_stack.plot_stimidx,u:(u+scl)]
        s2=out1['stim'][m.parent_stack.plot_stimidx,u:(u+scl)]
        pred, =plt.plot(s1,label='Pre-'+m.name)
        resp, =plt.plot(s2,'r',label='Post-'+m.name)
        plt.legend(handles=[pred,resp])
        plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx)+', trial #'+str(i))
        u=u+scl
"""
def non_plot(m):
    pass

def raster_data(data,pres,dura,posts,fr):
    s=data.shape
    pres=int(pres)
    dura=int(dura)
    posts=int(posts)
    xpre=np.zeros((s[2],pres*s[1]))
    ypre=np.zeros((s[2],pres*s[1]))
    xdur=np.zeros((s[2],dura*s[1]))
    ydur=np.zeros((s[2],dura*s[1]))
    xpost=np.zeros((s[2],posts*s[1]))
    ypost=np.zeros((s[2],posts*s[1]))
    for i in range(0,s[2]):
        spre=0
        sdur=0
        spost=0
        for j in range(0,s[1]):
            ypre[i,spre:(spre+pres)]=(j+1)*np.clip(data[:pres,j,i],0,1)
            xpre[i,spre:(spre+pres)]=np.divide(np.array(list(range(0,pres))),fr)
            ydur[i,sdur:(sdur+dura)]=(j+1)*np.clip(data[pres:(pres+dura),j,i],0,1)
            xdur[i,sdur:(sdur+dura)]=np.divide(np.array(list(range(pres,(pres+dura)))),fr)
            ypost[i,spost:(spost+posts)]=(j+1)*np.clip(data[(pres+dura):(pres+dura+posts),j,i],0,1)
            xpost[i,spost:(spost+posts)]=np.divide(
                    np.array(list(range((pres+dura),(pres+dura+posts)))),fr)
            spre+=pres
            sdur+=dura
            spost+=posts
    ypre[ypre==0]=None
    ydur[ydur==0]=None
    ypost[ypost==0]=None
    return(xpre,ypre,xdur,ydur,xpost,ypost)

def raster_plot(m,idx=None,size=(12,6)):
    """
    This function generates a raster plot of the data for the specified stimuli.
    It shows the spikes that occur during the actual trial in green, and the background
    spikes in grey. 
    """
    resp=m.parent_stack.unresampled['resp']
    pre=m.parent_stack.unresampled['prestim']
    dur=m.parent_stack.unresampled['duration']
    post=m.parent_stack.unresampled['poststim']
    freq=m.parent_stack.unresampled['respFs']
    reps=m.parent_stack.unresampled['repcount']
    ids=m.parent_stack.plot_stimidx
    r=reps.shape[0]
    prestim=float(pre)*freq
    duration=float(dur)*freq
    poststim=float(post)*freq
    if m.parent_stack.unresampled['pupil'] is not None:
        lis=[]
        for i in range(0,r):
            lis.extend([i]*reps[i])
        stims=lis[ids]
    else:
        stims=ids
    xpre,ypre,xdur,ydur,xpost,ypost=raster_data(resp,prestim,duration,poststim,freq)
    if idx is not None:
        plt.figure(num=str(stims)+str(100),figsize=size)
    plt.scatter(xpre[stims],ypre[stims],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xdur[stims],ydur[stims],color='g',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xpost[stims],ypost[stims],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.ylabel('Trial')
    plt.xlabel('Time')
    plt.title('Stimulus #'+str(stims))

def sorted_raster(m,idx=None,size=FIGSIZE):
    resp=m.parent_stack.unresampled['resp']
    pre=m.parent_stack.unresampled['prestim']
    dur=m.parent_stack.unresampled['duration']
    post=m.parent_stack.unresampled['poststim']
    freq=m.parent_stack.unresampled['respFs']
    reps=m.parent_stack.unresampled['repcount']
    r=reps.shape[0]
    prestim=float(pre)*freq
    duration=float(dur)*freq
    poststim=float(post)*freq
    pup=m.parent_stack.unresampled['pupil']
    idi=m.parent_stack.plot_stimidx
    lis=[]
    for i in range(0,r):
            lis.extend([i]*reps[i])
    ids=lis[idi]
    b=np.nanmean(pup[:,:,ids],axis=0)
    bc=np.asarray(sorted(zip(b,range(0,len(b)))),dtype=int)
    bc=bc[:,1]
    resp[:,:,ids]=resp[:,bc,ids]
    xpre,ypre,xdur,ydur,xpost,ypost=raster_data(resp,prestim,duration,poststim,freq)
    if idx is not None:
        plt.figure(num=str(ids)+str(100),figsize=size)
    plt.scatter(xpre[ids],ypre[ids],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xdur[ids],ydur[ids],color='g',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xpost[ids],ypost[ids],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.ylabel('Trial')
    plt.xlabel('Time')
    plt.title('Stimulus #'+str(ids))
    

#
# Other support functions
#

def shrinkage(mH,eH,sigrat=1,thresh=0):

    smd=np.abs(mH)/(eH+np.finfo(float).eps*(eH==0)) / sigrat

    if thresh:
       hf=mH*(smd>1)
    else:
       smd=1-np.power(smd,-2)
       smd=smd*(smd>0)
       #smd[np.isnan(smd)]=0
       hf=mH*smd
    
    return hf


