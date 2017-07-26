#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:16:37 2017

@author: svd
"""
import numpy as np
import nems.modules as nm
import nems.stack as ns
import nems.fitters as nf
import nems.utils as nu
import nems.keywords as nk
import nems.baphy_utils as baphy_utils
import os
import datetime
import copy
import scipy.stats as spstats

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

"""
fit_single_model - create, fit and save a model specified by cellid, batch and modelname

example fit on nice IC cell:
    import lib.main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    nems.fit_single_model(cellid,batch,modelname)

"""
def fit_single_model(cellid, batch, modelname, autoplot=True,**xvals): #Remove xvals later, need to rework web app
    """
    Fits a single NEMS model. With the exception of the autoplot feature,
    all the details of modelfitting are taken care of by the model keywords.
    
    fit_single_model functions by iterating through each of the keywords in the
    modelname, and perfroming the actions specified by each keyword, usually 
    appending a nems module. If nested crossvla is specified in one of the modules,
    it will simply re-evaluate the keywords for each new estimation set, and store
    the fitted parameters for each module. Once all the estimation sets have been fit,
    fit_single_model then evaulates the validation datasets and saves the stack.
    It autoplot is true, it generates plots specified by each nems module. 
    
    fit_single_model returns the evaluated stack, which contains both the estimation
    and validation datasets. In the caste of nested crossvalidation, the validation
    dataset contains all the data, while the estimation dataset is just the estimation 
    data that was fitted last (i.e. on the last nest)
    """
    stack=ns.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname
    
    # extract keywords from modelname    
    keywords=modelname.split("_")
    stack.cond=False
    while stack.cond is False:
        print('iter loop='+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in keywords:
            f = getattr(nk, k)
            f(stack)
            
       #TODO: this stuff below could be wrapped into do_fit somehow
        phi=[] 
        for idx,m in enumerate(stack.modules):
            this_phi=m.parms2phi()
            if this_phi.size:
                if stack.cv_counter==0:
                    stack.fitted_modules.append(idx)
                phi.append(this_phi)
        phi=np.concatenate(phi)
        stack.parm_fits.append(phi)

        if stack.nests==1:
            stack.cond=True
        else:
            stack.cv_counter+=1

    # measure performance on both estimation and validation data
    stack.valmode=True
    
    #stack.nests=1

    stack.evaluate(1)
    
    stack.append(nm.mean_square_error)
    
    corridx=nu.find_modules(stack,'correlation')
    if not corridx:
       # add MSE calculator module to stack if not there yet
        stack.append(nm.correlation)
                    
    #print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                 #stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
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
    
    
    # save
    filename=(
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl"
            .format(batch, cellid, modelname)
            )
    nu.save_model(stack,filename) 
    #os.chmod(filename, 0o666)

    return(stack)

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
def load_single_model(cellid, batch, modelname):
    
    filename=(
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl"
            .format(batch, cellid, modelname)
            )
    # For now don't do anything different to cross validated models.
    # TODO: should these be loaded differently in the future?
    #filename = filename.strip('_xval')
    
    stack=nu.load_model(filename)
    stack.evaluate()
    
    return stack
    