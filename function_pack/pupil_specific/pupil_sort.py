#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:50:49 2017

@author: shofer
"""
import numpy as np

#This extracts the trial number for trials where the pupil diameter was greater than some
#fraction of the maximum pupil diameter. In the future, it will actually sort all the 
#pupil data by pupil size, especially for raster plotting


def pupil_sort(data,obj=None,thresh=None,stimnum=0):
    s=data[:,:,stimnum].shape
    try:
        cutoff=obj.cutoff
    except:
        cutoff=thresh
    above=[]
    below=[]
    for j in range(0,s[1]):
        if np.nanmin(data[:,j,stimnum])>=cutoff:
            above.append(j)
        else:
            below.append(j)          
    return(above,below)