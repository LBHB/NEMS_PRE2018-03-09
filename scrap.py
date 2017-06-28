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



stack=nm.nems_stack()

stack.meta['batch']=294
stack.meta['cellid']='eno053f-a1_nat_export'


file='/auto/users/shofer/data/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'.mat'
print("Initializing load_mat with file {0}".format(file))
stack.append(nm.load_mat,est_files=[file],fs=100)
#print(stack.data[1]['repcount'])
stack.append(nm.pupil_est_val,valfrac=0.5)

alldata=stack.data






#stack.append(nm.standard_est_val,valfrac=0.05)
#stack.append(nm.dc_gain,g=1,d=0)
#stack.append(nm.sum_dim)



stack.append(nm.fir_filter,num_coefs=10)
#stack.append(nm.linpupgain)
stack.append(nm.mean_square_error)

stack.error=stack.modules[-1].error
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.05
stack.fitter.do_fit()
"""
stack.popmodule()
stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['dexp'])
stack.append(nm.mean_square_error)
stack.error=stack.modules[-1].error
                         
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.01
stack.fitter.do_fit()
"""
stack.quick_plot()

reps=stack.data[-1][0]['repcount']


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

stack.popmodule()


stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['dexp'])
stack.append(nm.mean_square_error)
stack.error=stack.modules[-1].error
                         
stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.0001
stack.fitter.do_fit()

stack.quick_plot()
"""





