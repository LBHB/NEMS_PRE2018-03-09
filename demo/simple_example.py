#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:29:12 2018

@author: svd
"""

import nems
import nems.modules as nm
import nems.utilities as nu
import nems.fitters as nf

import numpy as np

# set up metadata
#cellid="gus021d-a2"  # identifier for this data set
cellid="TAR010c-21-1"  # identifier for this data set
batch=0              # batch of data sets that this set belongs to
modelname="wc02_fir" # string identifier for this model architecture


# load and shape the data matrices appropriately
# PSEUDOCODE: data=my_load('stimulusfile','respfile')

stimfile="https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"+cellid+"_NAT_stim_ozgf_c18_fs100.mat"
respfile="https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"+cellid+"_NAT_resp_fs100.mat"

#use helper function: data=nu.io.load_matlab_file(respfile)
X=nu.io.load_matlab_matrix(stimfile,key="stim",label="stim",channelaxis=0,
                           eventaxis=1,timeaxis=2)
Y=nu.io.load_matlab_matrix(respfile,key="psth",label="resp",repaxis=0,
                           eventaxis=1,timeaxis=2)

# remove repetition axis
X=np.nanmean(X,axis=3)
Y=np.nanmean(Y,axis=3)

# first three events are validation data
X_est=X[:,3:,:]
Y_est=Y[:,3:,:]
X_val=X[:,0:2,:]
Y_val=Y[:,0:2,:]


# create and fit the model
# PSEDUOCODE: stack=fit_dumb_model(stim=X_est,resp=Y_est)

# intialize stack and data
stack=nems.stack.nems_stack(cellid=cellid,batch=batch,modelname=modelname)

stack.data=[[]]
stack.data[0].append({'stim': X_est, 'resp': Y_est, 'pred': X_est.copy(), 'respFs': 100, 'est': True})

# add model modules
stack.append(nm.filters.WeightChannels,num_chans=2)
stack.append(nm.filters.FIR,num_coefs=15)

# fit-related stuff
stack.append(nm.metrics.mean_square_error)
stack.error=stack.modules[-1].error


stack.fitter=nf.fitters.basic_min(stack)
stack.fitter.tolerance=1e-07
stack.fitter.do_fit()


# test validation data
# PSEUDOCODE r=test_dumb_model(stack=stack,stim=X_val,resp=Y_val)

stack.data[0].append({'stim': X_val, 'resp': Y_val, 'pred': X_val.copy(), 'respFs': 100, 'est': False})
stack.valmode=True
stack.evaluate(0)

stack.plot_dataidx=1  # plot example trial from val dataset
stack.quick_plot()


# TODO - save in JSON format with or without data matrix
# PSEUDOCODE: nu.io.save_dumb_model_as_json(stack)
