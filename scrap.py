#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:57:49 2017

@author: shofer
"""

import scipy.io
import scipy.signal
import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.baphy_utils as baphy_utils
import sys
sys.path.append('/auto/users/shofer/scikit-optimize')
import skopt.optimizer.gp as skgp
import copy


stack=nm.nems_stack()

stack.meta['batch']=294
stack.meta['cellid']='eno052b-a1_nat_export'
#stack.meta['batch']=283
#stack.meta['cellid']='BOL005c-07-1_nat_export'



file='/auto/users/shofer/data/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'.mat'
print("Initializing load_mat with file {0}".format(file))
stack.append(nm.load_mat,est_files=[file],fs=100,formpup=False)
#print(stack.data[1]['repcount'])
#stack.append(nm.pupil_est_val,valfrac=0.5)
alldats=copy.deepcopy(stack.data)
stack.append(nm.pupil_model,tile_data=True)

alldata=stack.data

#stack.append(nm.fir_filter,num_coefs=10)

#stack.append(nm.mean_square_error)

#stack.error=stack.modules[-1].error
                         

#stack.fitter=nf.basic_min(stack)

#stack.fitter.tol=0.1
#stack.fitter.do_fit()

#stack.popmodule()
stack.append(nm.linpupgain)
#stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['dexp'])
stack.append(nm.mean_square_error)
stack.error=stack.modules[-1].error
                         
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.01
stack.fitter.do_fit()

stack.plot_trialidx=(10,14)
stack.trial_quick_plot()

reps=stack.data[-1][0]['repcount']
unres=stack.unresampled

"""

stack=nm.nems_stack()

stack.meta['batch']=291
stack.meta['cellid']='bbl031f-a1'

#stack.append(nm.dummy_data,data_len=200)
file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
print("Initializing load_mat with file {0}".format(file))
stack.append(nm.load_mat,est_files=[file],fs=100)
stack.append(nm.standard_est_val,valfrac=0.5)

alldat=stack.data

#stack.append(nm.dc_gain,g=1,d=0)
#stack.append(nm.sum_dim)
stack.append(nm.fir_filter,num_coefs=10)
stack.append(nm.mean_square_error)

stack.error=stack.modules[-1].error
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.005
stack.fitter.do_fit()

a=[0.1]*181
b=[-0.1]*181
c=list(zip(b,a))


stack.fitter=nf.forest_min(stack,dims=c)
stack.fitter.do_fit()
#stack.popmodule()


#stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['dexp'])
#stack.append(nm.mean_square_error)
#stack.error=stack.modules[-1].error
                         
#stack.fitter=nf.basic_min(stack)
#stack.fitter.tol=0.0001
#stack.fitter.do_fit()

stack.quick_plot()


"""



