#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example model fit using NEMS and some basic calls to baphy-specific files and
celldb data.  This should not live in demo/ forever, but it's meant to provide
a helpful parallel/contrast with the celldb & baphy-indepedent examples.

As in simple_example.py, the model is a linear-nonlinear STRF, with the
linear filter constrained to be rank-2 and the output nonlinearity defined
as a double-exponential (see Thorson et al. PLoS CB 2015 for details)

Created on Mon Jan 15 09:29:12 2018

@author: svd
"""

import nems.stack as ns
import nems.modules as nm
import nems.utilities as nu
import nems.fitters as nf
import nems.db as nd

import numpy as np

# set up metadata
use_example="RDT"

if use_example=="NAT":
    # This is A1/Natural sound example
    #cellid="gus021d-a2"      # identifier for this data set
    cellid="TAR010c-21-2"     # identifier for this data set
    batch=271                 # batch of data sets that this set belongs to
    modelname="wc02_fir_dexp" # string identifier for this model architecture
    valeventcount=3

elif use_example=="RDT":
    # this is an RDT example:
    cellid="chn029d-a1"
    batch=269                 # batch of data sets that this set belongs to
    modelname="wc02_fir_dexp" # string identifier for this model architecture
    valeventcount=20

# get file info from database
d=nd.get_batch_cell_data(batch,cellid)

stim_options={'filtfmt': 'ozgf', 'fsout': 100, 'chancount': 18, 'includeprestim': 1}
stimfile=nu.baphy.stim_cache_filename(d['stim'][0],stim_options)

resp_options={'rasterfs': 100,'includeprestim': 1}
respfile=nu.baphy.spike_cache_filename2(d['raster'][0],resp_options)

#use helper function: data=nu.io.load_matlab_file(respfile)
# returns stimulus, X = [(spectral)channel X event X time]
# and response, Y = [(neural)channel X event X time]
# event and time axis sizes must match for the two matrices
X=nu.io.load_matlab_matrix(stimfile,key="stim",label="stim",channelaxis=0,
                           eventaxis=2,timeaxis=1)
Y=nu.io.load_matlab_matrix(respfile,key="r",label="resp",repaxis=1,
                           eventaxis=2,timeaxis=0)

# because of idiosyncratic baphy behavior, having to do with allowing more 
# reps in the validation segment, some "events" will never actually be
# played, (0 reps, all nans in Y). This will remove them. Only relevant for 
# NAT data, not RDT
Y=np.nanmean(Y,axis=3)
keepidx=np.isfinite(Y[0,:,0])
Y=Y[:,keepidx,:]
X=X[:,keepidx,:]


# FROM HERE DOWN, THIS SHOULD BE IDENTICAL TO simple_example.py

# first three events are validation data, separate them out
X_est=X[:,valeventcount:,:]
Y_est=Y[:,valeventcount:,:]
X_val=X[:,0:valeventcount,:]
Y_val=Y[:,0:valeventcount,:]


# create and fit the model
# PSEDUOCODE: stack=fit_LN_model(stim=X_est,resp=Y_est)

# intialize stack and data
stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)

stack.data=[[]]
stack.data[0].append({'stim': X_est, 'resp': Y_est, 'pred': X_est.copy(), 'respFs': 100, 'est': True})

# add model modules
stack.append(nm.filters.WeightChannels,num_chans=2)
stack.append(nm.filters.FIR,num_coefs=15)
nu.utils.mini_fit(stack)     # fit linear model some before adding static NL

stack.append(nm.nonlin.gain, nltype='dexp')
nu.utils.mini_fit(stack,['nonlin.gain'])    # fit only static NL

# set up the cost function
stack.append(nm.metrics.mean_square_error)
stack.error=stack.modules[-1].error

# set up the fitter
stack.fitter=nf.fitters.basic_min(stack)
stack.fitter.tolerance=1e-06

# run the final fit
stack.fitter.do_fit()


# test validation data
# PSEUDOCODE r=test_dumb_model(stack=stack,stim=X_val,resp=Y_val)

stack.append(nm.metrics.correlation)
stack.data[0].append({'stim': X_val, 'resp': Y_val, 'pred': X_val.copy(), 'respFs': 100, 'est': False})
stack.valmode=True
stack.evaluate(0)

stack.plot_dataidx=1  # plot example trial from val dataset
stack.quick_plot()


# TODO - save in JSON format with or without data matrix
# PSEUDOCODE: nu.io.save_dumb_model_as_json(stack)
#
# open questions/options
# 1. save only metadata and fit parameters
# 2. save metadata, fit parameters and input signals
