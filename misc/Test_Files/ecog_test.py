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
import nems.utilities as ut
import nems.stack as ns

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import h5py

imp.reload(nm)
imp.reload(main)
imp.reload(nf)
imp.reload(nk)
imp.reload(ut)
imp.reload(ns)


channel=1
cellid="sam-{0:03d}".format(channel)
batch=300 #ECOG

modelname="ecog25_wcg01_fir10_fit01"

if 0:
    stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)
else:
    stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)
    stack.valmode=False
    stack.keyfuns=nk.keyfuns
        
    # evaluate the stack of keywords    
    if 'nest' in stack.keywords[-1]:
        # special case if last keyword contains "nested". TODO: better imp!
        print('Evaluating stack using nested cross validation. May be slow!')
        k=stack.keywords[-1]
        stack.keyfuns[k](stack)
    else:
        print('Evaluating stack')
        for k in stack.keywords:
            stack.keyfuns[k](stack)

    if 1:
        # validation stuff
        stack.valmode=True
        stack.evaluate(0)
        
        stack.append(nm.metrics.correlation)
        
        #print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
        #             stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
        valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
        if valdata:
            stack.plot_dataidx=valdata[0]
        else:
            stack.plot_dataidx=0

    stack.modules[0].do_plot=ut.plot.plot_spectrogram
    stack.quick_plot()
    
    if 0:
        filename = ut.io.get_file_name(cellid, batch, modelname)
        ut.io.save_model(stack, filename)

