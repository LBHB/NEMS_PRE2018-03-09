#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:15:26 2017

@author: shofer
"""

import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.baphy_utils as bu
import lib.nems_modules as nm
import lib.nems_fitters as nf
import sys

import copy


stack=nm.nems_stack()

stack.meta['batch']=294
stack.meta['cellid']='eno048e-a1'

#file='/auto/users/shofer/data/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'.mat'
file=bu.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
print("Initializing load_mat with file {0}".format(file))

stack.append(nm.load_mat,est_files=[file],fs=100,formpup=False)
stack.append(nm.pupil_est_val,valfrac=0.05)
stack.append(nm.normalize)

stack.append(nm.pupil_model,tile_data=True)

#smalldata=copy.deepcopy(stack.data)

#stack.append(nm.state_gain,gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0])
#stack.append(nm.state_gain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,0,1])
#stack.append(nm.state_gain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25],order=2)
stack.append(nm.state_gain,gain_type='exppupgain',fit_fields=['theta'],theta=[0,1,0,0])
#stack.append(nm.state_gain,gain_type='logpupgain',fit_fields=['theta'],theta=[0,0,0,1])
#stack.append(nm.state_gain,gain_type='Poissonpupgain',fit_fields=['theta'],theta=[10,20])

#stack.append(nm.pseudo_huber_error,b=0.3)
stack.append(nm.mean_square_error)

stack.error=stack.modules[-1].error
                         
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.000001
stack.fitter.do_fit()
"""
stack.popmodule()
stack.append(nm.mean_square_error)
print(stack.modules[-1].mse_est)
"""

alldata=copy.deepcopy(stack.data)
reps=stack.data[1][0]['repcount']
#smalldata=copy.deepcopy(stack.data)
unres=stack.unresampled
stack.plot_stimidx=150 #Choose which stimulus to plot
#stack.plot_trialidx=(10,11) #Choose which trials to display

#print(stack.modules[3].theta)                   
#stack.do_sorted_raster()
#stack.trial_quick_plot()
stack.quick_plot()
#resout=stack.do_sorted_raster()


