#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:04:00 2017

@author: shofer
"""
import numpy as np
import matplotlib.pyplot as plt


def pred_vs_resp(data,obj=None,stims='all',trials=0,size=(12,4),**kwargs):
    preds=data['predicted']
    resps=data['resp']
    sr=resps.shape
    if obj is not None:
        scale=obj.newHz
    else:
        scale=kwargs['frequency']
    xlist=np.arange(0,sr[0])
    xlist=xlist/scale
    if stims=='all':
        ran=range(0,sr[2])
    elif isinstance(stims,int):
        ran=range(stims,stims+1)
    else:
        ran=range(stims[0],stims[1]+1)
    for i in ran:
        if isinstance(trials,int):
            plt.figure((str(i)+str(trials)),figsize=size)
            plt.plot(xlist,preds[:,trials,i])
            plt.plot(xlist,resps[:,trials,i],'g')
            plt.ylabel('Firing Rate')
            plt.xlabel('Time (s)')
            plt.title('Stimulus #'+str(i)+', Trial #'+str(trials))
        else:
            for j in range(trials[0],trials[1]+1):
                plt.figure((str(i)+str(j)),figsize=size)
                plt.plot(xlist,preds[:,j,i])
                plt.plot(xlist,resps[:,j,i],'g')
                plt.ylabel('Firing Rate')
                plt.xlabel('Time (s)')
                plt.title('Stimulus #'+str(i)+', Trial #'+str(j))
                
           
def pred_vs_avgresp(data,obj=None,stims='all',size=(12,4),**kwargs):
    preds=data['predicted']
    resps=data['resp']
    sr=resps.shape
    if obj is not None:
        scale=obj.newHz
    else:
        scale=kwargs['frequency']
    xlist=np.arange(0,sr[0])
    xlist=xlist/scale
    respavg=np.nanmean(resps,axis=1)
    predavg=np.nanmean(preds,axis=1)
    if stims=='all':
        ran=range(0,sr[2])
    elif isinstance(stims,int):
        ran=range(stims,stims+1)
    else:
        ran=range(stims[0],stims[1]+1)
    for i in ran:
        plt.figure(i,figsize=(12,4))
        plt.plot(xlist,predavg[:,i])
        plt.plot(xlist,respavg[:,i],'g')
        plt.ylabel('Firing Rate')
        plt.xlabel('Time (s)')
        plt.title('Stimulus #'+str(i))
            


