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


def load_ecog(channel=0):
    
    stimfile='/auto/data/daq/ecog/coch.mat'
    respfile='/auto/data/daq/ecog/reliability0.1.mat'
    
    stimdata = h5py.File(stimfile,'r')
    respdata = h5py.File(respfile,'r')
    
    data={}
    for name,d in respdata.items():
        #print (name)
        data[name]=d.value
    for name,d in stimdata.items():
        #print (name)
        data[name]=d.value
    data['resp']=data['D'][channel,:,:]   # shape to stim X time (25Hz)
    
    # reshape stimulus to be channel X stim X time and downsample from 400 to 25 Hz
    stim_resamp_factor=int(400/25)
    noise_thresh=0
    # reduce spectral sampling to speed things up
    data['stim']=ut.utils.thresh_resamp(data['coch_all'],6,thresh=noise_thresh,ax=1)
    
    # match temporal sampling to response
    data['stim']=ut.utils.thresh_resamp(data['stim'],stim_resamp_factor,thresh=noise_thresh,ax=2)
    data['stim']=np.transpose(data['stim'],[1,0,2])

    data['repcount']=np.ones([data['resp'].shape[0],1])
    data['pred']=data['stim']
    
    return data


channel=1
cellid='sam-001'
batch=300 #ECOG

modelname="xval10_wcg01_fir10_fit01"

if 0:
    stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)
else:
    d=load_ecog(channel)
    stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)
    stack.valmode=False
    stack.keyfuns=nk.keyfuns
    stack.data=[]
    stack.data.append([])
    stack.data[0].append(d)
    
    # extract keywords from modelname, look up relevant functions in nk and save
    # so they don't have to be found again.
    
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
        stack.evaluate(1)
        
        stack.append(nm.metrics.correlation)
        
        #print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
        #             stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
        valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
        if valdata:
            stack.plot_dataidx=valdata[0]
        else:
            stack.plot_dataidx=0

    #nlidx=nems.utilities.utils.find_modules(stack,'nonlin.gain')
    #stack.modules[nlidx[0]].do_plot=nems.utilities.utils.io_scatter_smooth
    stack.modules[0].do_plot=ut.plot.plot_spectrogram
    stack.quick_plot()
    
    if 0:
        filename = ut.io.get_file_name(cellid, batch, modelname)
        ut.io.save_model(stack, filename)

