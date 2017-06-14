#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:16:31 2017

@author: shofer
"""

import numpy as np
import scipy.signal as sps
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
        #y=sps.fftconvolve(X[i,:],obj.coeffs[i,:]) #possibly faster for large arrays
        X[i,:]=y[0:X.shape[1]]
    X=X.sum(0)#+obj.base
    output=np.reshape(X,s[1:])
    obj.current=output
    return(output)