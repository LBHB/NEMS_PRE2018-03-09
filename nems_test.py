#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import numpy as np
import matplotlib as mp
import pylab as pl

import scipy.io
import scipy.signal
from nems_mod import *

datapath='/Users/svd/python/nems/ref/week5_TORCs/'

est_files=[datapath + 'tor_data_por073b-b1.mat']

stack=nems_stack()
stack.append(load_mat(est_files=est_files))
stack.append()   # pass-through module

stack.eval()
out1=stack.output()
print('stim[0][0]: {0}'.format(out1[0]['stim'][0][0]))


stack.append(fir_filter(d_in=out,num_coefs=10))
stack.modules[-1].coefs[0,0]=1
stack.modules[-1].coefs[0,2]=2
stack.eval(1)
out2=stack.output()
print('stim[0][0]: {0}'.format(out2[0]['stim'][0][0]))

stack.append(add_scalar(n=2))
stack.eval(1)
out3=stack.output()
print('stim[0][0]: {0}'.format(out3[0]['stim'][0][0]))

d_in=stack.data[1]  # same as out1?

pl.figure()
ax=pl.subplot(2,1,1);
ax.imshow(out1[0]['stim'][:,0,:], aspect='auto', origin='lower')
#ax.imshow(out1[0]['stim'][:,:,0], interpolation='nearest', aspect='auto',origin='lower')
ax=pl.subplot(2,1,2);
ax.plot(out3[0]['stim'][0,:])
ax.plot(out3[0]['resp'][0,:],'r')


