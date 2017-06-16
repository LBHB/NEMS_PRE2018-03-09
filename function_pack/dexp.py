#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:27:05 2017

@author: shofer
"""
import numpy as np

def create_dexp(obj,**kwargs):
    obj.dexp=np.ones([1,4]) 
    obj.dexp[0][1]=0
    obj.dexp[0][3]=0 
    return(['dexp'])
    print('dexp parameters created')


def DEXP(obj,**kwargs):
    ins=kwargs['pred'] #data should be obj.current
    v1=obj.dexp[0,0]
    v2=obj.dexp[0,1]
    v3=obj.dexp[0,2]
    v4=obj.dexp[0,3]
    output=v1-v2*np.exp(-np.exp(v3*(ins-v4)))
    obj.current=output
    return(output)