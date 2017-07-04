#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:15:26 2017

@author: shofer
"""

import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.baphy_utils as baphy_utils
import lib.nems_modules as nm
import lib.nems_fitters as nf
import sys

import copy


stack=nm.nems_stack()

stack.meta['batch']=294
stack.meta['cellid']='eno024d-b2_nat_export'

file='/Users/HAL-9000/Desktop/CompNeuro/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'.mat'
print("Initializing load_mat with file {0}".format(file))

stack.append(nm.load_mat,est_files=[file],fs=100,formpup=False)
stack.append(nm.pupil_model,tile_data=True)
#stack.append(nm.state_gain,gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0])
stack.append(nm.state_gain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,0,1])
#stack.append(nm.state_gain,gain_type='exppupgain',fit_fields=['theta'],theta=[0,1,0,0])
#stack.append(nm.state_gain,gain_type='logpupgain',fit_fields=['theta'],theta=[0,0,0,1])
stack.append(nm.mean_square_error)

stack.error=stack.modules[-1].error
                         
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.001
stack.fitter.do_fit()

alldata=stack.data

stack.plot_stimidx=0 #Choose which stimulus to plot
stack.plot_trialidx=(10,11) #Choose which trials to display

stack.do_sorted_raster()
stack.trial_quick_plot()
#resout=stack.do_sorted_raster()