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
    while stack.cond is False:
        print('iter loop='+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in keywords:
            f = getattr(nk, k)
            f(stack)
            
        fitted_modules=[]
        for idx,m in enumerate(self.stack.modules):
            this_phi=m.parms2phi()
            if this_phi.size:
                fitted_modules.append(idx)
            phi=[]
            for k in fitted_modules:
                g=stack.modules[k].parms2phi()
                phi=np.append(phi,g)
        stack.cv_counter+=1
            
        # measure performance on both estimation and validation data
    stack.valmode=True
    stack.evaluate(1)
    corridx=nu.find_modules(stack,'correlation')
    if not corridx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.correlation)
                
    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
             stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    if stack.cross_val is not True:
        stack.cond=True
        valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
        if valdata:
            stack.plot_dataidx=valdata[0]
        else:
            stack.plot_dataidx=0
    else:
        
        
        
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
    if stack.cross_val is not True:
        return(stack)
    else:
        #TODO: Figure out best way to output data
        E=0
        P=0
        val_stim=np.concatenate(val_stim_list,axis=0)
        val_resp=np.concatenate(val_resp_list,axis=0)
        
        E=np.sum(np.square(val_stim-val_resp))
        P=np.sum(np.square(val_resp))
        mse=E/P
        stack.meta['mse_val']=mse
        #stack.meta['mse_val']=np.median(np.array(mse_vallist))
        stack.meta['mse_est']=np.median(np.array(mse_estlist))
        stack.meta['r_est']=np.median(np.array(r_est_list))
        val_stim=val_stim.reshape([-1,1],order='C')
        val_resp=val_resp.reshape([-1,1],order='C')
        
        stack.meta['r_val'],p=spstats.pearsonr(val_stim,val_resp)
        #stack.meta['r_val']=np.median(np.array(r_val_list))
        print("Median: mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
              stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
        #return(stack_list)
        return(stack)