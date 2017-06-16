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
import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk

#datapath='/Users/svd/python/nems/ref/week5_TORCs/'
#est_files=[datapath + 'tor_data_por073b-b1.mat']

#datapath='/auto/data/code/nems_in_cache/batch271/'
#est_files=[datapath + 'chn020f-b1_b271_ozgf_c24_fs200.mat']
datapath='/Users/svd/python/nems/misc/ref/'
est_files=[datapath + 'bbl031f-a1_nat_export.mat']

# create an empty stack
stack=nm.nems_stack()

#stack.meta['cellid']='chn020f-b1'
stack.meta['cellid']='bbl031f-a1'
stack.meta['batch']=291

# add a loader module to stack
#stack.append(nm.load_mat(est_files=est_files,fs=100))
#nk.fb24ch200(stack)
nk.fb18ch100(stack)
stack.append(nm.standard_est_val())

# add fir filter module to stack
nk.fir10(stack)

# add MSE calculator module to stack
stack.append(nm.mean_square_error())

# set error (for minimization) for this stack to be output of last module
stack.error=stack.modules[-1].error
stack.eval(1)

stack.fitter=nf.nems_fitter(stack)
#stack.fitter=nf.basic_min(stack)
#stack.fitter.maxit=500

stack.fitter.do_fit()

stack.popmodule()
stack.append(nm.dexp())
stack.append(nm.mean_square_error())

stack.fitter=nf.basic_min(stack)
stack.fitter.maxit=100  # does this work??
stack.fitter.do_fit()

# display the output of each
pl.figure()
for idx,m in enumerate(stack.modules):
    ax=pl.subplot(5,1,idx+1)
    stack.modules[idx].do_plot(ax)
    


## old display stuff
#out3=stack.output()
#mse=stack.modules[-1].output
#print('mse: {0}'.format(mse))
##print('stim[0][0]: {0}'.format(out3[0,0]))
##
#d_in=stack.data[1]  # same as out1?
#
#pl.set_cmap('jet')
#pl.figure()
#ax=pl.subplot(2,2,1)
#ax.imshow(out1[0]['stim'][:,0,:], aspect='auto', origin='lower')
##ax.imshow(out1[0]['stim'][:,:,0], interpolation='nearest', aspect='auto',origin='lower')
#
#ax=pl.subplot(2,2,2)
#ax.imshow(stack.modules[1].coefs,aspect='auto',origin='lower')
#
#ax=pl.subplot(2,2,3);
#ax.plot(out3[0]['stim'][0,:])
#ax.plot(out3[0]['resp'][0,:],'r')


