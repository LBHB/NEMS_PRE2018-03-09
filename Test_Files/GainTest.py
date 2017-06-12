#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:18:51 2017

@author: shofer
"""

import numpy as np
import copy
import math as mt
import scipy as sp
import scipy.signal as sps
import scipy.io as si
import matplotlib as mp
import matplotlib.pyplot as plt
import os
import os.path
from newNEM import *

'''
filelist=os.listdir('/auto/users/shofer/data/batch294')
ratiolist=[]
for i in filelist:
    val=0
    val=FERReT(batch=294,cellid=i,n_coeffs=10,thresh=0.4).pupil_comparison()
    ratiolist.append(val)
mean=np.nanmean(np.array(ratiolist))
'''

val2=FERReT(batch=294,cellid='eno048f-a1_nat_export.mat',n_coeffs=10,thresh=0.4).raster_plot()