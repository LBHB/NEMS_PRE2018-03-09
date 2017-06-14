#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017

@author: svd
"""

import scipy
import numpy as np

def load_baphy_file(f):

    matdata = scipy.io.loadmat(f,chars_as_strings=True)
    s=matdata['data'][0][0]
    data={} 
    data['resp']=s['resp_raster']   
    data['stim']=s['stim']
    data['respFs']=s['respfs']
    data['stimFs']=s['stimfs']
    data['stimparam']=[str(''.join(letter)) for letter in s['fn_param']]
    data['isolation']=s['isolation']
    data['tags']=s['tags']   
    
    return data

