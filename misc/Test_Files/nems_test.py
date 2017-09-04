#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import imp
import scipy.io
import pkgutil as pk

import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities.utils as nu
import nems.stack as ns

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

imp.reload(nm)
imp.reload(main)
imp.reload(nf)
imp.reload(nk)
imp.reload(nu)
imp.reload(ns)

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
cellid='BOL006b-11-1'
batch=293
modelname="parm50_wc01_fir15_fititer00"

#cellid='bbl034e-a1'
cellid='bbl031f-a1'
batch=291
cellid='zee015e-04-1'
batch=271
#modelname="fb18ch100_wc01_stp1pc_fir15_dexp_fititer00"
#modelname="fb18ch100_wc01_fir15_dexp_fititer00"
#modelname="fb18ch100_wc01_fir15_dexp_fititer00"
#modelname="fb18ch100_wc01_stp2pc_fir15_dexp"
modelname="fb18ch100_wcg01_fir15_dexp_fit01"
#cellid="eno052d-a1"
#batch=294
#modelname="perfectpupil50_pupgain_fit01"

# pupil gain test
#cellid="BOL006b-11-1"
#cellid="eno053d-c3"
#batch=293
#modelname="parm50_wc01_fir15_pupwgtctl_dexp_fit01_nested10"


#cellid="eno023c-c1"
#batch=294
#modelname="perfectpupil50_pupgain_fit01_nested5"

cellid='gus018d-d1'
#cellid="gus023e-c2"
batch=296
#modelname="env100e_stp1pc_fir20_fit01"
modelname="env100e_fir20_dexp_fit01"

# following is equivalent of 
#stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)

if 0:
    stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)
else:
    stack=ns.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname
    stack.valmode=False
    
    # extract keywords from modelname, look up relevant functions in nk and save
    # so they don't have to be found again.
    stack.keywords=modelname.split("_")
    """
    stack.keyfun={}
    for k in stack.keywords:
        f=None
        for importer, modname, ispkg in pk.iter_modules(nk.__path__):
            try:
                f=getattr(importer.find_module(modname).load_module(modname),k)
                break
            except:
                pass
        if f is None:
            raise ValueError("Keyword {0} not found.".format(k))
        stack.keyfun[k]=f
    """
    
    # evaluate the stack of keywords    
    if 'nested' in stack.keywords[-1]:
        # special case if last keyword contains "nested". TODO: better imp!
        print('Evaluating stack using nested cross validation. May be slow!')
        k=stack.keywords[-1]
        nk.keyfuns[k](stack)
    else:
        print('Evaluating stack')
        for k in stack.keywords:
            nk.keyfuns[k](stack)

if 1:
    # validation stuff
    stack.valmode=True
    stack.evaluate(1)
    
    stack.append(nm.metrics.correlation)
                    
    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                 stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx=valdata[0]
    else:
        stack.plot_dataidx=0

#nlidx=nu.find_modules(stack,'nonlin.gain')
#stack.modules[nlidx[0]].do_plot=nu.io_scatter_smooth
stack.quick_plot()



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
