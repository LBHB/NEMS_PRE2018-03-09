#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:16:32 2017

@author: shofer
"""


import numpy as np

def create_pupgain(obj,**kwargs):
    obj.pupgain=np.zeros([1,4])
    obj.pupgain[0][1]=1
    return(['pupgain'])
    print('pupgain parameters created')

def pupgain(obj,**kwargs):
    ins=kwargs['pred'] 
    pups=kwargs['pupdata']
    d0=obj.pupgain[0,0]
    g0=obj.pupgain[0,1]
    d=obj.pupgain[0,2]
    g=obj.pupgain[0,3]
    output=d0+(d*pups)+(g0*ins)+g*np.multiply(pups,ins)
    obj.current=output
    return(output)