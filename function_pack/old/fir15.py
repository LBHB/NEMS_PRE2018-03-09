#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:16:31 2017

@author: shofer
"""

import numpy as np
#import scipy.signal as sps


def create_fir15(obj,**kwargs):
    obj.fir15=np.zeros([obj.dims,15])
    obj.model='fir15'
    #obj.base=np.zeros([1,1])
    return(['fir15'])
    print('fir15 parameters created')
    
    
def fir15(obj,**kwargs): 
    X=kwargs['data']
    s=X.shape
    X=np.reshape(X,[s[0],-1])
    for i in range(0,s[0]):
        y=np.convolve(X[i,:],obj.fir15[i,:])
        #y=sps.fftconvolve(X[i,:],obj.coeffs[i,:]) #possibly faster for large arrays
        X[i,:]=y[0:X.shape[1]]
    X=X.sum(0)#+obj.base
    output=np.reshape(X,s[1:])
    obj.current=output
    return(output)