#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:27:05 2017

@author: shofer
"""
import numpy as np

def create_DEXP(obj,**kwargs):
    obj.DEXP=np.ones([1,4]) 
    obj.DEXP[0][1]=0
    obj.DEXP[0][3]=0 
    return(['DEXP'])
    print('DEXP parameters created')


def DEXP(obj,**kwargs):
    ins=kwargs['data'] #data should be obj.current
    v1=obj.DEXP[0,0]
    v2=obj.DEXP[0,1]
    v3=obj.DEXP[0,2]
    v4=obj.DEXP[0,3]
    output=v1-v2*np.exp(-np.exp(v3*(ins-v4)))
    obj.current=output
    return(output)