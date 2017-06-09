#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:16:32 2017

@author: shofer
"""


import numpy as np


def pupil_gain(obj,**kwargs):
    #ins=copy.deepcopy(self.pred)
    ins=kwargs['data'] #data should be self.pred
    #pups=copy.deepcopy(self.train['pup'])
    pups=kwargs['pupdata']
    d0=obj.pupil[0,0]
    g0=obj.pupil[0,1]
    d=obj.pupil[0,2]
    g=obj.pupil[0,3]
    output=d0+(d*pups)+(g0*ins)+g*np.multiply(pups,ins)
    obj.current=output
    return(output)