#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_utils as nu
import lib.baphy_utils as baphy_utils

# loader keywords
def fb24ch200(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat(est_files=[file],fs=200))

def fb18ch100(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat(est_files=[file],fs=100))

def loadlocal(stack):
    file='/auto/users/shofer/data/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'.mat'
    #file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat(est_files=[file],fs=100))

# fir filter keywords
def fir10(stack):
    out1=stack.output()
    stack.append(nm.fir_filter(d_in=out1,num_coefs=10))

def fir15(stack):
    out1=stack.output()
    stack.append(nm.fir_filter(d_in=out1,num_coefs=15))


# static NL keywords
def dlog(stack):
    out1=stack.output()
    stack.append(nm.nonlinearity(d_in=out1,nltype='dlog',fit_params=['dlog']))
    
def exp(stack):
    out1=stack.output()
    stack.append(nm.nonlinearity(d_in=out1,nltype='exp',fit_params=['exp']))

def dexp(stack):
    stack.append(nm.dexp)


# fitter keywords
def fit00(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error())

        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
        
    stack.fitter=nf.simplex()
    stack.fit()

# etc etc for other keywords
    
    