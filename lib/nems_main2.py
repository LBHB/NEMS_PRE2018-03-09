#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:10:22 2017

@author: HAL-9000
"""
import numpy as np
import lib.nems_modules as nm
import lib.nems_stack as ns
import lib.nems_fitters as nf
import lib.nems_utils as nu
import lib.nems_keywords as nk
import lib.baphy_utils as baphy_utils
import os
import datetime
import copy
import scipy.stats as spstats

def fit_single_model(cellid, batch, modelname, autoplot=True,crossval=False):
    """
    Fits a single NEMS model.
    
    Crossval should be working now! At least for pupil stuff ---njs July 13 2017
    """
    stack=ns.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname
    stack.cross_val=crossval
    
    # extract keywords from modelname    
    keywords=modelname.split("_")
    stack.cv_counter=0
    stack.cond=False
    stack.fitted_modules=[]
    while stack.cond is False:
        print('iter loop='+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in keywords:
            f = getattr(nk, k)
            f(stack)
            
        phi=[]
        for idx,m in enumerate(stack.modules):
            this_phi=m.parms2phi()
            if this_phi.size:
                if stack.cv_counter==0:
                    stack.fitted_modules.append(idx)
                phi.append(this_phi)
        phi=np.concatenate(phi)
        stack.parm_fits.append(phi)
        if stack.cross_val is not True:
            stack.cond=True
        
        stack.cv_counter+=1
            
        # measure performance on both estimation and validation data
    stack.valmode=True
    if stack.cross_val is True:
        stack.nested_evaluate(1)
        stack.nested_concatenate(1)
    else:
        stack.evaluate(1)
    corridx=nu.find_modules(stack,'correlation')
    if not corridx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.correlation)
                
    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
             stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx=valdata[0]
    else:
        stack.plot_dataidx=0

    # edit: added autoplot kwarg for option to disable auto plotting
    #       -jacob, 6/20/17
    if autoplot:
        stack.quick_plot()
    
    # add tag to end of modelname if crossvalidated
    if crossval:
        # took tag out for now, realized it would cause issues with loader.
        # TODO: how should load model handle the tag? Or don't bother wih tag?
        xval = ""
        #xval = "_xval"
    else:
        xval = ""
    
    # save
    filename=(
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}{3}.pkl"
            .format(batch, cellid, modelname, xval)
            )
    #nu.save_model(stack,filename) 
    #os.chmod(filename, 0o666)

    return(stack)