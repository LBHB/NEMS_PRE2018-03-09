#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:53:43 2017

@author: shofer
"""
import numpy as np
import math as mt

def create_gauss(obj,**kwargs):
    obj.gauss=np.ones([1,3])
    return(['gauss'])
    print('gauss parameters created')


def gauss(obj,**kwargs):
    ins=kwargs['pred']
    mu=obj.gauss[0,0]
    sigma=obj.gauss[0,1]
    offset=obj.gauss[0,2]
    output=(1/(sigma*mt.sqrt(2*mt.pi)))*np.exp(-(np.square(ins-mu)/(2*(sigma^2))))+offset
    obj.current=output
    return(output)