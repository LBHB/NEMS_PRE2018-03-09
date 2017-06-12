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

import Plotting.raster_plot as rp
import Plotting.coeff_heatmap as ch
import Plotting.comparison as pc

test3=FERReT(batch=294,cellid='eno048f-a1_nat_export.mat',n_coeffs=15,queue=('input_log','FIR','pupil_gain'),thresh=0.4)
print(test3.input_log)
print(test3.coeffs.shape)
print(test3.pupil)
'''
data3=test3.ins['resp']
pre_time3=test3.meta['prestim']
dur_time3=test3.meta['duration']
post_time3=test3.meta['poststim']
frequency3=test3.meta['respf']


rp.raster_plot(stims='all',size=(6,9),data=data3,pre_time=pre_time3,dur_time=dur_time3,
               post_time=post_time3,frequency=frequency3)
'''
#test3.run_fit(validation=0.5,reps=1)

test3.data_resample(noise_thresh=0.04)
test3.reshape_repetitions()
test3.create_datasets(valsize=0.5)
test3.input_log=np.load('/auto/users/shofer/Saved_Data/test3log.npy')
test3.coeffs=np.load('/auto/users/shofer/Saved_Data/test3coeffs.npy')
test3.pupil=np.load('/auto/users/shofer/Saved_Data/test3pups.npy')
#ch.heatmap(coeffs=coeffs3)
#plt.figure(num=10,figsize=(15,5))
#plt.plot(test3.current[275:550])
#plt.plot(test3.train['resp'][275:550])
valid=test3.apply_to_val(save=False,filepath='/auto/users/shofer/Saved_Data/valtest3.npy')

pc.pred_vs_resp(obj=test3,data=valid,stims='all',trials=(5,10))




#np.save('/auto/users/shofer/Saved_Data/test3log.npy',test3.input_log)
#np.save('/auto/users/shofer/Saved_Data/test3coeffs.npy',test3.coeffs)
#np.save('/auto/users/shofer/Saved_Data/test3pups.npy',test3.pupil)