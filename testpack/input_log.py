#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:20:13 2017

@author: shofer
"""
import numpy as np

def create_input_log(obj,**kwargs):
    obj.input_log=np.ones([1,1])
    return(['input_log'])
    print('input_log parameters created')
     

def input_log(obj,**kwargs):
    X=kwargs['indata'] #Once gammatone filter is created, change to ['data']
    v1=obj.input_log[0,0]
    output=np.log(X+v1)
    obj.current=output
    return(output)