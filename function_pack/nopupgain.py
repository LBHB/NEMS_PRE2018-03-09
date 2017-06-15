#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:20:34 2017

@author: shofer
"""
import numpy as np

def create_nopupgain(obj,**kwargs):
    obj.nopupgain=np.zeros([1,2])
    obj.nopupgain[0][1]=1
    return(['nopupgain'])
    print('nopupgain parameters created')
    
    
def nopupgain(obj,**kwargs):
    ins=kwargs['pred'] #data should be self.pred
    d0=obj.nopupgain[0,0]
    g0=obj.nopupgain[0,1]
    output=d0+(g0*ins)
    obj.current=output
    return(output)