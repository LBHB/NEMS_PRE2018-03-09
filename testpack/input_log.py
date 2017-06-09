#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:20:13 2017

@author: shofer
"""
import numpy as np

def input_log(obj,data):
    #X=copy.deepcopy(self.train['stim'])
    X=data
    v1=obj.log[0,0]
    output=np.log(X+v1)
    obj.train['stim']=output
    return(output)