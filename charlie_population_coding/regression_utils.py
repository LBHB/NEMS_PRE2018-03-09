#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:24:33 2018

@author: hellerc
"""


import statsmodels.api as sm
import pandas as pd
import numpy as np
def linear_model(r, p):
    
    if len(r.shape)==4:
        r = r.reshape(r.shape[0]*r.shape[1]*r.shape[2],r.shape[3])
        p = p.reshape(p.shape[0]*p.shape[1]*p.shape[2])
        p = pd.DataFrame(index=range(0,len(p)),columns=['pupil'],data=p)
        p= sm.add_constant(p)
    elif len(r.shape)==3:
        r = r.reshape(r.shape[0]*r.shape[1],r.shape[2])
        p = p.reshape(p.shape[0]*p.shape[1])
        p = pd.DataFrame(index=range(0,len(p)),columns=['pupil'],data=p)
        p= sm.add_constant(p)
    pred = np.empty(r.shape)
    rsq=[]
    
    for i in range(0, r.shape[1]):
        y=pd.DataFrame(index=range(0,len(r[:,i])),columns=['neuron'], data =r[:,i])
        lnmodel=sm.OLS(y, p)
        results=lnmodel.fit()
        rsq.append(results.rsquared);
        pred[:,i] = results.predict();
    
    
    return (pred, rsq)