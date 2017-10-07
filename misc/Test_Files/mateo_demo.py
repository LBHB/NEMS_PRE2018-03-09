#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:18:15 2017

@author: shofer
"""

import imp
import scipy.io

import nems.main as main
import nems.modules as nm
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as nu
import nems.stack as ns

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

imp.reload(main)
imp.reload(nm)
imp.reload(nf)
imp.reload(nk)
imp.reload(nu)
imp.reload(ns)



batch=296
#cellid='gus018d-d1'
cellid='gus023e-c2'
modelname="env100e_fir20_fit01_ssa"
#modelname="env100e_stp1pc_fir20_fit01"


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
    
    # set to zero to skip validation test
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

# OVERRIDE default plot for mean_square_error mo/home/mateo/nems/untitled0.pydule
mse_idx=nu.utils.find_modules(stack,'metrics.mean_square_error')
stack.modules[mse_idx[0]].do_plot=nu.plot.pred_act_psth_smooth

edge_idx=nu.utils.find_modules(stack,'aux.onset_edges')
stack.modules[edge_idx[0]].do_plot=nu.plot.plot_spectrogram


stack.plot_dataidx=0
stack.plot_stimidx=1

stack.quick_plot()
