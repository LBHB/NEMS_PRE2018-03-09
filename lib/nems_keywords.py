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
    stack.append(nm.load_mat,est_files=[file],fs=200)
    
def fb24ch100(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100) #Data not preprocessed to 100 Hz, internally converts
    
def fb18ch100(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100)

def loadlocal(stack):
    """
    This keyword is just to load up a local file that is not yet on the BAPHY database.
    Right now just loads files from my computer --njs, June 27 2017
    """
    file='/auto/users/shofer/data/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'.mat'
    #file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100)
    

standard_est_val = ['ev', ]

def ev(stack):
    stack.append(nm.standard_est_val, valfrac=0.05)


# weight channels keywords
###############################################################################

def wc01(stack):
    stack.append(nm.weight_channels,num_chans=1)

def wc02(stack):
    stack.append(nm.weight_channels,num_chans=2)

def wc03(stack):
    stack.append(nm.weight_channels,num_chans=3)

def wc04(stack):
    stack.append(nm.weight_channels,num_chans=4)


# fir filter keywords
###############################################################################

def fir_mini_fit(stack):
    # mini fit
    stack.append(nm.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.fitter=nf.basic_min(stack)
    stack.fitter.tol=0.05
    #stack.fitter=nf.coordinate_descent(stack)
    #stack.fitter.tol=0.001
    fitidx=nu.find_modules(stack,'weight_channels') + nu.find_modules(stack,'fir_filter')
    stack.fitter.fit_modules=fitidx
    
    stack.fitter.do_fit()
    stack.popmodule()
    
    
def fir10(stack):
    stack.append(nm.fir_filter,num_coefs=10)
    fir_mini_fit(stack)
    
    
def fir15(stack):
    stack.append(nm.fir_filter,num_coefs=15)
    fir_mini_fit(stack)

# static NL keywords
###############################################################################

def dlog(stack):
    stack.append(nm.nonlinearity,nltype='dlog',fit_fields=['phi'],phi=[1])
    
def exp(stack):
    stack.append(nm.nonlinearity,nltype='exp',fit_fields=['phi'],phi=[1,1])

def dexp(stack):
    stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['phi'],phi=[1,1,1,1])
    
def poly01(stack):
    stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1])
    
def poly02(stack):
    stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1,0])
    
def poly03(stack):
    stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1,0,0])


# state variable keyowrds
###############################################################################

def nopupgain(stack):
    stack.append(nm.state_gain,gain_type='nopupgain',fit_fields=['theta'],theta=[0,1])
    
def pupgain(stack):
    stack.append(nm.state_gain,gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0])

def polypupgain04(stack): #4th degree polynomial gain fn
    stack.append(nm.state_gain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,0,0,1])
    
def polypupgain03(stack): #3rd degree polynomial gain fn
    stack.append(nm.state_gain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,0,1])
    
def exppupgain(stack):
    stack.append(nm.state_gain,gain_type='exppupgain',fit_fields=['theta'],theta=[0,1,0,0])

def logpupgain(stack):
    stack.append(nm.state_gain,gain_type='logpupgain',fit_fields=['theta'],theta=[0,1,0,1])

# fitter keywords
###############################################################################

def fit00(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
        
    stack.evaluate(1)

    stack.fitter=nf.basic_min(stack)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    
def fitannl00(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
    
    stack.evaluate(1)
    
    stack.fitter=nf.anneal_min(stack,anneal_iter=50,stop=5,up_int=10,bounds=None)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    

# etc etc for other keywords
###############################################################################

def perfectpupil(stack):
    """keyword to fit pupil gain using "perfect" model generated by pupil_model module.
    The idea here is not to fit a model to the data, but merely to see the effect of a 
    a linear pupil gain function on the "perfect" model generated by averaging the
    rasters of each trial for a given stimulus. This keyword loads up the data
    and generates the model. It should be used with pupgain and a fitter keyword.
    """
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,formpup=False)
    stack.append(nm.pupil_est_val,valfrac=0)
    stack.append(nm.pupil_model,tile_data=True)
    
    
    
