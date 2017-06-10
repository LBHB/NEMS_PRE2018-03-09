#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:16:31 2017

@author: shofer
"""

import numpy as np
#import copy

def create_FIR(obj,**kwargs):
    obj.coeffs=np.zeros([obj.dims,obj.n_coeffs])
    obj.base=np.zeros([1,1])
    return(['coeffs'])
    print('FIR parameters created')
    
    
def FIR(obj,**kwargs): 
    X=kwargs['data']
    s=X.shape
    X=np.reshape(X,[s[0],-1])
    for i in range(0,s[0]):
        y=np.convolve(X[i,:],obj.coeffs[i,:])
        X[i,:]=y[0:X.shape[1]]
    X=X.sum(0)#+obj.base
    obj.current=np.reshape(X,s[1:])
    return(obj.current)