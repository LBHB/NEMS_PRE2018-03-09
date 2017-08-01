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
    k=pred_vs_resp.count
    for i in ran:
        if isinstance(trials,int):
            plt.figure((str(k)+str(i)+str(trials)),figsize=size)
            plt.plot(xlist,preds[:,trials,i])
            plt.plot(xlist,resps[:,trials,i],'r')
            plt.ylabel('Firing Rate')
            plt.xlabel('Time (s)')
            plt.title('Stimulus #'+str(i)+', Trial #'+str(trials))
        else:
            for j in range(trials[0],trials[1]+1):
                plt.figure((str(k)+str(i)+str(j)),figsize=size)
                plt.plot(xlist,preds[:,j,i])
                plt.plot(xlist,resps[:,j,i],'r')
                plt.ylabel('Firing Rate')
                plt.xlabel('Time (s)')
                plt.title('Stimulus #'+str(i)+', Trial #'+str(j))
    pred_vs_resp.count+=1    
           
def pred_vs_avgresp(data,obj=None,stims='all',size=(12,4),**kwargs):
    preds=data['predicted']
    resps=data['resp']
    if obj is not None:
        scale=obj.newHz
    else:
        scale=kwargs['frequency']
    if data['pup'] is not None:
        respavg=np.nanmean(resps,axis=1)
        predavg=np.nanmean(preds,axis=1)
    else:
        respavg=resps
        predavg=preds
    sr=respavg.shape
    xlist=np.arange(0,sr[0])
    xlist=xlist/scale
    if stims=='all':
        ran=range(0,sr[1])
    elif isinstance(stims,int):
        ran=range(stims,stims+1)
    else:
        ran=range(stims[0],stims[1]+1)
    k=pred_vs_avgresp.count
    for i in ran:
        plt.figure((str(k)+str(i)),figsize=(12,4))
        plt.plot(xlist,predavg[:,i])
        plt.plot(xlist,respavg[:,i],'r')
        plt.ylabel('Firing Rate')
        plt.xlabel('Time (s)')
        plt.title('Stimulus #'+str(i))
    pred_vs_avgresp.count+=1
            


