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
from nems_modules import *

#datapath='/Users/svd/python/nems/ref/week5_TORCs/'
#est_files=[datapath + 'tor_data_por073b-b1.mat']

datapath='/auto/data/code/nems_in_cache/batch271/'
est_files=[datapath + 'chn020f-b1_b271_ozgf_c24_fs200.mat']

# create an empty stack
stack=nems_stack()

# add a loader module to stack
stack.append(load_mat(est_files=est_files,fs=100))
stack.eval()
out1=stack.output()
#print('stim[0][0]: {0}'.format(out1[0]['stim'][0][0]))

# add fir filter module to stack
stack.append(fir_filter(d_in=out1,num_coefs=10))
stack.eval(1)
out2=stack.output()
#print('stim[0][0]: {0}'.format(out2[0]['stim'][0][0]))

# add MSE calculator module to stack
stack.append(mean_square_error())

# set error (for minimization) for this stack to be output of last module
stack.error=stack.modules[-1].error
stack.eval(1)

# pull out current phi as initial conditions
phi0=stack.modules[1].parms2phi()

# create fitter, this should be turned into an object in the nems_fitters libarry
def test_cost(phi):
    stack.modules[1].phi2parms(phi)
    stack.eval(1)
    test_cost.counter+=1
    if test_cost.counter % 100 == 0:
        print('Eval #{0}. MSE={1}'.format(test_cost.counter,stack.error()))
    return stack.meta['est_mse']
    
test_cost.counter=0

# run the fitter
phi=scipy.optimize.fmin(test_cost, phi0, maxiter=1000)


# display the output of each
pl.figure()
ii=0
ax=pl.subplot(3,1,1)
stack.modules[0].do_plot(ax)
ax=pl.subplot(3,1,2)
stack.modules[1].do_plot(ax)
ax=pl.subplot(3,1,3)
stack.modules[2].do_plot(ax)


# old display stuff
out3=stack.output()
mse=stack.modules[-1].output
print('mse: {0}'.format(mse))
#print('stim[0][0]: {0}'.format(out3[0,0]))
#
d_in=stack.data[1]  # same as out1?

pl.set_cmap('jet')
pl.figure()
ax=pl.subplot(2,2,1)
ax.imshow(out1[0]['stim'][:,0,:], aspect='auto', origin='lower')
#ax.imshow(out1[0]['stim'][:,:,0], interpolation='nearest', aspect='auto',origin='lower')

ax=pl.subplot(2,2,2)
ax.imshow(stack.modules[1].coefs,aspect='auto',origin='lower')

ax=pl.subplot(2,2,3);
ax.plot(out3[0]['stim'][0,:])
ax.plot(out3[0]['resp'][0,:],'r')


