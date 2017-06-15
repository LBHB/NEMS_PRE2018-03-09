#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 00:31:43 2017

@author: svd
"""


# two pseudocode examples of usage


# pseudocode option 1 (command line execution):
    
import nems_modules as nm
import nems_fitters as nf
import pylab as pl

cellid="chn004b-a1"
batch=271
est_file="/auto/data/code/nems_in_cache/batch271/chn004b-a1_b271_ozgf_c24_fs200.mat"

m=FERReT()
m.cellid=cellid
m.batch=batch

m.infile=est_file
m.initialize_files() # load the file(s)
m.create_datasets(0.05)  # maybe included in previous function?

# add_module function something like nems_stack.append() in nems_mod.py
m.add_module(nm.nonlinearity('log_compress'))
m.add_module(nm.FIR(15))  # where 15 is the number of time bins
m.add_module(nm.nonlinearity('dexp'))
m.add_module(nm.MSE)

m.fitter=nf.simplex()

m.fit()

pl.figure()
ii=0
for mod in m.stack():
    ax=pl.subplot(4,1,ii++)
    mod.do_plot(ax)
    


# pseudocode option 2 (run through keyword list):
import nems_modules as nm    # nems_modules is something like nems_mod.py of old
import nems_fitters as nf    # collection of fit algorithms
import nems_keyword as nk    # collection of functions that evaluate keywords

cellid="chn004b-a1"
batch=271
keywords=['fb24ch100','lognn','fir15','dexp','fit00']

m=FERReT()
m.cellid=cellid
m.batch=batch

for k in keywords:
    method_to_call = getattr(nk, k)
    method_to_call(m)
    
m.generate_summary_plot()

m.save_results_to_file()



# contents of nems_keyword.py
import nems_modules as nm
import nems_fitters as nf
import baphy_utils

def fb24ch100(m):
    file=baphy_utils.get_celldb_file(m.batch,m.cellid,200,'ozgf',18)
    m.infile=file
    m.initialize_files()
    m.create_datasets()  # maybe included in previous function?

def lognn(m):
    m.addmodule(nm.nonlinearity('log_compress'))

def fir15(m):
    m.addmodule(nm.FIR(15))  # where 15 is the number of time bins

def dexp(m):
    m.addmodule(nm.nonlinearity('dexp'))

def fit00(m):
    m.fitter=nf.simplex())
    m.fit()

# etc etc for other keywords



