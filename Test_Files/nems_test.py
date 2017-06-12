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

#datapath='/Users/svd/python/nems/ref/week5_TORCs/'
#est_files=[datapath + 'tor_data_por073b-b1.mat']

datapath='/home/svd/python/nems/ref/'
est_files=[datapath + 'bbl031f-a1_nat_export.mat']

stack=nems_stack()

stack.append(load_mat(est_files=est_files,fs=100))
stack.eval()
out1=stack.output()
#print('stim[0][0]: {0}'.format(out1[0]['stim'][0][0]))

stack.append(fir_filter(d_in=out1,num_coefs=10))
stack.eval(1)
out2=stack.output()
#print('stim[0][0]: {0}'.format(out2[0]['stim'][0][0]))

stack.append(mean_square_error())
stack.error=stack.modules[-1].error
stack.eval(1)

phi0=stack.modules[1].parms2phi()

def test_cost(phi):
    stack.modules[1].phi2parms(phi)
    stack.eval(1)
    test_cost.counter+=1
    if test_cost.counter % 100 == 0:
        print('Eval #{0}. MSE={1}'.format(test_cost.counter,stack.error()))
    return stack.meta['est_mse']
    
test_cost.counter=0

phi=scipy.optimize.fmin(test_cost, phi0)


out3=stack.output()
mse=stack.modules[-1].output
print('mse: {0}'.format(mse))
#print('stim[0][0]: {0}'.format(out3[0,0]))
#
d_in=stack.data[1]  # same as out1?

pl.set_cmap('jet')
pl.figure()
ax=pl.subplot(2,2,1);
ax.imshow(out1[0]['stim'][:,0,:], aspect='auto', origin='lower')
#ax.imshow(out1[0]['stim'][:,:,0], interpolation='nearest', aspect='auto',origin='lower')

ax=pl.subplot(2,2,2)
ax.imshow(stack.modules[1].coefs,aspect='auto',origin='lower')

ax=pl.subplot(2,2,3);
ax.plot(out3[0]['stim'][0,:])
ax.plot(out3[0]['resp'][0,:],'r')


