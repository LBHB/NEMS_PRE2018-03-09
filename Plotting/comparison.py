#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:04:00 2017

@author: shofer
"""
import numpy as np
import matplotlib.pyplot as plt


def pred_vs_resp(obj,data,stims='all',trials=0,size=(15,5)):
    preds=data['predicted']
    resps=data['resp']
    sr=resps.shape
    if stims=='all':
        ran=range(0,sr[2])
    elif isinstance(stims,int):
        ran=range(stims,stims+1)
    else:
        ran=range(stims[0],stims[1]+1)
    for i in ran:
        if isinstance(trials,int):
            plt.figure((str(i)+str(trials)),figsize=size)
            plt.plot(preds[:,trials,i])
            plt.plot(resps[:,trials,i],'g')
            plt.ylabel('Firing Rate')
            plt.xlabel('Time Step')
            plt.title('Stimulus #'+str(i)+', Trial #'+str(trials))
        else:
            for j in range(trials[0],trials[1]+1):
                plt.figure((str(i)+str(j)),figsize=size)
                plt.plot(preds[:,j,i])
                plt.plot(resps[:,j,i],'g')
                plt.ylabel('Firing Rate')
                plt.xlabel('Time Step')
                plt.title('Stimulus #'+str(i)+', Trial #'+str(j))
                
"""              
def pred_vs_avgresp(obj,data,stims='all'):
    
    
    
            if trials is False:
            plt.figure(i,figsize=(12,4))
            plt.plot(preds[:,i])
            plt.plot(resps[:,i],'g')
            plt.ylabel('Firing Rate')
            plt.xlabel('Time Step')
            plt.title('Stimulus #'+str(i))
            
            
"""