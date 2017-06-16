#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:36:00 2017

@author: shofer
"""

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import math as mt
import scipy.io
import scipy.signal
from nems_mod2 import *
from newNEM import *
import copy
def avg_trials_test(data):
    avgs=np.nanmean(data,axis=1)
    return(avgs)


test2=FERReT(batch=294,cellid='eno048f-a1_nat_export.mat',n_coeffs=20,thresh=0.4)

#val2=test2.pupil_comparison()

#test2.raster_plot(stims=(0,1))

#FERReT.new_method=test_fn

#test2.new_method() #Adding an externally defined function
#However, it is probably better to just define a subclass that inherits from FERReT
#to add more functions

queue=('FIR','pupil_gain','pupil_gain') 
#The first element in a queue should always be a modeling function, whether
#it's the FIR filter, factorized filter, parametrized filter, or some user 
#defined model. The first element in the queue is called first, and translates 
#the input stimulus data into a "prediction". The subsequent order of the modules
#does not matter as far as program functionality, but may effect fitting performance

print(getattr(test2,'mse'))
test2.run_fit(queue=queue,validation=0.5,reps=2)
test2.heatmap(model='FIR')
#assem=test2.assemble(avgresp=False,useval=True,
                                  #save=False,filepath='/auto/users/shofer/code/nems/Saved_Data/test2.npy')
#assem=np.load('/auto/users/shofer/Saved_Data/test2.npy')[()]
#test2.plot_pred_resp(assem,stims='all',trials=(20,30))


#plt.figure(4,figsize=(15,5))
#plt.plot(assem['predicted'][:,25,0])
#plt.plot(assem['resp'][:,25,0],'g')










