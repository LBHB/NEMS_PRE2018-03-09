#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:07:21 2017

@author: shofer
"""
import numpy as np

#This module sorts response data into two arrays based on a threshold pupil 
#diameter. This is useful for plotting two PSTH's to observe any obvious 
#difference in response that corresponds to increased or decreaased pupil diameter

#should be input a 3-D array of responses formatted as (T,R,S)

def state_PSTH_data(data,above,below,stimnum=0):
    resp=data[:,:,stimnum]
    high=[]
    low=[]
    la=len(above)
    lb=len(below)
    for i in range(0,la):
        j=above[i]
        high.append(resp[:,j])
    for i in range(0,lb):
        j=below[i]
        low.append(resp[:,j])
    high=np.array(high)
    high=np.transpose(high)
    low=np.array(low)
    low=np.transpose(low)
    return(high,low)