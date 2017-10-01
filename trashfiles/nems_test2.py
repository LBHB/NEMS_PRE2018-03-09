#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:57:12 2017

@author: svd
"""

import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('/auto/users/shofer/nems') #TODO: You may want to comment this out
import scipy.io
import scipy.signal
import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.baphy_utils as baphy_utils


# create an empty stack
stack=nm.nems_stack()

stack.meta['batch']=291
stack.meta['cellid']='bbl031f-a1'

#stack.append(nm.dummy_data,data_len=200)
file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
print("Initializing load_mat with file {0}".format(file))
stack.append(nm.load_mat,est_files=[file],fs=100)
stack.append(nm.standard_est_val,valfrac=0.05)

#stack.append(nm.dc_gain,g=1,d=0)
#stack.append(nm.sum_dim)
stack.append(nm.nonlinearity,nltype='dlog',fit_fields=['dlog'],phi0=[1],premodel=True)
stack.append(nm.fir_filter,num_coefs=10)
stack.append(nm.pseudo_huber,b=0.5)

stack.error=stack.modules[-1].error
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.01
stack.fitter.do_fit()

stack.popmodule()
stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['dexp'],phi0=[1,1,1,1])
stack.append(nm.pseudo_huber,b=0.5)
stack.error=stack.modules[-1].error
                         
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.01
out=stack.fitter.do_fit()

alldata=stack.data

stack.quick_plot()