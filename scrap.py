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
import lib.baphy_utils as bu
import sys
sys.path.append('/auto/users/shofer/scikit-optimize')
import skopt.optimizer.gp as skgp
import copy

"""
stack=nm.nems_stack()

stack.meta['batch']=267
stack.meta['cellid']='mag009a-c1'




file=bu.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
print("Initializing load_mat with file {0}".format(file))
stack.append(nm.load_mat,est_files=[file],fs=100,formpup=False)
stack.append(nm.standard_est_val,valfrac=0.5)
stack.append(nm.normalize)
alldats=copy.deepcopy(stack.data)




stack.append(nm.weight_channels,num_chans=3)
#stack.append(nm.nonlinearity,nltype='dlog',fit_fields=['phi'],phi=[1])
stack.append(nm.fir_filter,num_coefs=15)

stack.append(nm.mean_square_error)
stack.error=stack.modules[-1].error
                      
stack.fitter=nf.basic_min(stack)

stack.fitter.tol=0.001
stack.fitter.do_fit()

stack.popmodule()

stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['phi'],phi=[1,1,1,1])
stack.append(nm.mean_square_error)
stack.error=stack.modules[-1].error
                       
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.001
stack.fitter.do_fit()

stack.fitter=nf.fit_by_type(stack,min_kwargs={'basic_min':{'routine':'L-BFGS-B','maxit':10000},'anneal_min':
            {'min_method':'L-BFGS-B','anneal_iter':50,'stop':10,'maxiter':10000,'up_int':5,'bounds':None,
                'temp':0.1,'stepsize':0.15,'verb':False}})
stack.fitter.tol=0.0001
stack.fitter.do_fit()
"""
stack.plot_stimidx=20
stack.quick_plot(size=(9,18))

reps=stack.data[-1][0]['repcount']
unres=stack.unresampled




