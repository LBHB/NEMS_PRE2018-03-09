#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:20:34 2017

@author: shofer
"""
import numpy as np

def create_no_pupil_gain(obj,**kwargs):
    obj.no_pupil=np.zeros([1,2])
    obj.no_pupil[0][1]=1
    return(['no_pupil'])
    print('no_pupil_gain parameters created')
    
    
def no_pupil_gain(obj,**kwargs):
    ins=kwargs['pred'] #data should be self.pred
    d0=obj.no_pupil[0,0]
    g0=obj.no_pupil[0,1]
    output=d0+(g0*ins)
    obj.current=output
    return(output)