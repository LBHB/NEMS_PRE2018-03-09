#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:41:50 2017

@author: shofer

This function generates a heatmap for model coefficients. This is probably most
useful for FIR or factorized spectrogram models.
"""

import matplotlib.pyplot as plt

def heatmap(obj=None,module=None,size=(6,6),**kwargs):
    if obj is not None:
        arr=getattr(obj,obj.fit_param[module][0])
    else:
        arr=kwargs['coeffs']
    plt.figure(figsize=size)
    plt.imshow(arr)
    plt.colorbar()