#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import nems_modules as nm
import nems_fitters as nf
import nems_utils as nu
import baphy_utils

def fb24ch200(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat(est_files=[file],fs=200))

def lognn(stack):
    print("lognn not implemented")
    #stack.addmodule(nm.nonlinearity('log_compress'))
    
    
def fir15(stack):
    stack.addmodule(nm.fir_filter(15))  # where 15 is the number of time bins

def dexp(stack):
    stack.addmodule(nm.dexp)

def fit00(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        stack.append(nm.mean_square_error())
        
    stack.fitter=nf.simplex()
    stack.fit()

# etc etc for other keywords
    
    