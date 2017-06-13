#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:40:55 2017

@author: shofer
"""
import numpy as np

def create_tanhsig(obj,**kwargs):
    obj.tanhsig=np.ones([1,3])
    obj.tanhsig[0][2]=0
    return(['tanhsig'])
    print('tanh_sig parameters created')
    

def tanhsig(obj,**kwargs):
    ins=kwargs['pred'] #data should be self.pred
    v1=obj.tanhsig[0,0]
    v2=obj.tanhsig[0,1]
    v3=obj.tanhsig[0,2]
    output=v1*np.tanh(v2*ins-v3)+v1
    obj.current=output
    return(output)