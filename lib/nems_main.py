#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:16:37 2017

@author: svd
"""

import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_utils as nu
import lib.nems_keywords as nk
import lib.baphy_utils as baphy_utils


"""
fit_single_model - create, fit and save a model specified by cellid, batch and modelname

example fit on nice IC cell:
    import lib.nems_main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    nems.fit_single_model(cellid,batch,modelname)

"""
def fit_single_model(cellid, batch, modelname, autoplot=True):
    
    stack=nm.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname

    # extract keywords from modelname    
    keywords=modelname.split("_")
    
    # evaluate each keyword in order
    for k in keywords:
        f = getattr(nk, k)
        f(stack)

    # measure performance on both estimation and validation data
    stack.valmode=True
    stack.evaluate(1)
    corridx=nu.find_modules(stack,'correlation')
    if not corridx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.correlation)
    
    print("Final r_est={0} r_val={1}".format(stack.meta['r_est'],stack.meta['r_val']))
    
    # default results plot, show validation data if exists
    valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx=valdata[0]
    else:
        stack.plot_dataidx=0
        
    # edit: added autoplot kwarg for option to disable auto plotting
    #       -jacob, 6/20/17
    if autoplot:
        stack.quick_plot()
    
    # save
    filename="/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl".format(batch,cellid,modelname)
    nu.save_model(stack,filename)
    return stack

"""
load_single_model - load and evaluate a model, specified by cellid, batch and modelname

example:
    import lib.nems_main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    stack=nems.load_single_model(cellid,batch,modelname)
    stack.quick_plot()
    
"""
def load_single_model(cellid,batch,modelname):
    
    filename="/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl".format(batch,cellid,modelname)
    stack=nu.load_model(filename)
    stack.evaluate()
    
    return stack
    
