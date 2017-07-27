#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import numpy as np
import matplotlib.pyplot as plt
import imp

import scipy.io
import scipy.signal
import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keywords as nk
import nems.utils as nu
import nems.stack as ns

#imp.reload(nf)

#datapath='/Users/svd/python/nems/ref/week5_TORCs/'
#est_files=[datapath + 'tor_data_por073b-b1.mat']

#datapath='/auto/data/code/nems_in_cache/batch271/'
#est_fieles=[datapath + 'chn020f-b1_b271_ozgf_c24_fs200.mat']
#datapath='/Users/svd/python/nems/misc/ref/'
#est_files=[datapath + 'bbl031f-a1_nat_export.mat']
#'/auto/users/shofer/data/batch291/bbl038f-a2_nat_export.mat'
# create an empty stack
cellid='eno052b-c1'
batch=293
modelname="parm100_xval05_wc02_fir15_fititer00"

stack=main.fit_single_model(cellid, batch, modelname)


'''

stack=ns.nems_stack()

stack.meta['batch']=291
#stack.meta['cellid']='chn020f-b1'
#stack.meta['cellid']='bbl031f-a1'
stack.meta['cellid']='bbl061h-a1'
#stack.meta['cellid']='bbl038f-a2_nat_export'

#stack.meta['batch']=267
#stack.meta['cellid']='ama024a-21-1'
stack.meta['batch']=293
stack.meta['cellid']='eno052b-c1'


# add a loader module to stack
#nk.fb18ch100(stack)
nk.parm100(stack)
#nk.loadlocal(stack)

#nk.ev(stack)
stack.append(nm.crossval, valfrac=0.00)

# add fir filter module to stack & fit a little
#nk.dlog(stack)
#stack.append(nm.normalize)
#nk.dlog(stack)
nk.wc02(stack)
nk.fir15(stack)

# add nonlinearity and refit
#nk.dexp(stack)

# following has been moved to nk.fit00
stack.append(nm.mean_square_error,shrink=0.5)
stack.error=stack.modules[-1].error


stack.fitter=nf.fit_iteratively(stack,max_iter=5)
#stack.fitter.sub_fitter=nf.basic_min(stack)
stack.fitter.sub_fitter=nf.coordinate_descent(stack,tol=0.001,maxit=10)
stack.fitter.sub_fitter.step_init=0.05

stack.fitter.do_fit()

stack.valmode=True
stack.evaluate(1)
corridx=nu.find_modules(stack,'correlation')
if not corridx:
    # add MSE calculator module to stack if not there yet
    stack.append(nm.correlation)    

stack.plot_dataidx=1

# default results plot
stack.quick_plot()

# save
#filename="/auto/data/code/nems_saved_models/batch{0}/{1}.pkl".format(stack.meta['batch'],stack.meta['cellid'])
#nu.save_model(stack,filename)


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
    
'''