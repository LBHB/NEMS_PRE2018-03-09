#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:06:46 2017

@author: shofer



"""
import numpy as np
import matplotlib.pyplot as plt


#TODO: Want to add "sorted-raster" function that produces a raster plot where the
#the trials are sorted by pupil_diameter
   
def raster_plot(obj=None,stims='all',size=(12,6),**kwargs):
    """
    This function generates a raster plot of the data for the specified stimuli.
    It shows the spikes that occur during the actual trial in green, and the background
    spikes in grey. 
    
    Can be called either using a FERReT object, or by manually inputing keyworded data:
        data=obj.ins['resp'] (response raster)
        pre_time=obj.meta['prestim'] (prestim silence time)
        dur_time=obj.meta['duration'] (stimulus duration)
        post_time=obj.meta['poststim'] (poststim silence)
        frequency=obj.meta['respf'] (sampling frequency)
    """
    if obj is not None:
        ins=obj.ins['resp']
        pre=obj.meta['prestim']
        dur=obj.meta['duration']
        post=obj.meta['poststim']
        freq=obj.meta['respf']
    else:
        ins=kwargs['data']
        pre=kwargs['pre_time']
        dur=kwargs['dur_time']
        post=kwargs['post_time']
        freq=kwargs['frequency']
    prestim=float(pre)*freq
    duration=float(dur)*freq
    poststim=float(post)*freq
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
    xpre,ypre,xdur,ydur,xpost,ypost=raster_data(ins,prestim,duration,poststim,freq)
    ran=[]
    rs=xpre.shape
    if stims=='all':
        ran=range(0,rs[0])
    elif isinstance(stims,int):
        ran=range(stims,stims+1)
    else:
        ran=range(stims[0],stims[1]+1)
    for i in ran:
        plt.figure(i,figsize=size)
        plt.scatter(xpre[i],ypre[i],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
        plt.scatter(xdur[i],ydur[i],color='g',s=(0.5*np.pi)*2,alpha=0.6)
        plt.scatter(xpost[i],ypost[i],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
        plt.ylabel('Trial')
        plt.xlabel('Time')
        plt.title('Stimulus #'+str(i))