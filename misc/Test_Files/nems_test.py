#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.io
import scipy.signal
import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk
import lib.nems_utils as nu

#datapath='/Users/svd/python/nems/ref/week5_TORCs/'
#est_files=[datapath + 'tor_data_por073b-b1.mat']

#datapath='/auto/data/code/nems_in_cache/batch271/'
#est_files=[datapath + 'chn020f-b1_b271_ozgf_c24_fs200.mat']
datapath='/Users/svd/python/nems/misc/ref/'
est_files=[datapath + 'bbl031f-a1_nat_export.mat']
#'/auto/users/shofer/data/batch291/bbl038f-a2_nat_export.mat'
# create an empty stack
stack=nm.nems_stack()

stack.meta['batch']=291
#stack.meta['cellid']='chn020f-b1'
stack.meta['cellid']='bbl031f-a1'
#stack.meta['cellid']='bbl038f-a2_nat_export'

# add a loader module to stack
nk.fb18ch100(stack)
#nk.loadlocal(stack)

stack.append(nm.standard_est_val)

# add fir filter module to stack
#nk.dlog(stack)
nk.fir10(stack)

## add MSE calculator module to stack
stack.append(nm.mean_square_error)
#
## set error (for minimization) for this stack to be output of last module
stack.error=stack.modules[-1].error
stack.evaluate(1)
#
#nk.fit00(stack)
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.05

stack.fitter.do_fit()

mse=stack.modules[-1].error()
print('mse after stage 1: {0}'.format(mse))

# add nonlinearity and refit
stack.popmodule()
nk.dexp(stack)
stack.append(nm.mean_square_error)

stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.005
stack.fitter.do_fit()

stack.quick_plot()
## single figure display
#plt.figure(figsize=(8,9))
#for idx,m in enumerate(stack.modules):
#    plt.subplot(len(stack.modules),1,idx+1)
#    m.do_plot()
    
## display the output of each module in a separate figure
#for idx,m in enumerate(stack.modules):
#    plt.figure(num=idx,figsize=(8,3))
#    #ax=plt.plot(5,1,idx+1)
#    m.do_plot(idx=idx)
    

