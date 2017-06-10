#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:15:37 2017

@author: shofer
"""

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import math as mt
import scipy.io
import scipy.signal
from nems_mod2 import *
from modNEM import *
import copy



test3=FERReT(batch=294,cellid='eno048f-a1_nat_export.mat',n_coeffs=15,queue=('input_log','FIR','pupil_gain'),thresh=0.4)
print(test3.input_log)
print(test3.coeffs.shape)
print(test3.pupil)

test3.run_fit(validation=0.5,reps=1)
#
#test3.basic_min(['pupil_gain'])
#test3.testtest()
#queue=('FIR','pupil_gain','pupil_gain') 
#test3.run_fit(validation=0.5,reps=2)
#test3.heatmap(model='FIR')
