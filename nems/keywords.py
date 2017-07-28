#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import nems.modules as nm
import nems.fitters as nf
import nems.utils as nu
import nems.baphy_utils as baphy_utils
import numpy as np

#thismod=sys.modules(__name__)

# loader keywords
def parm100(stack):
    """
    Specifically for batch293 tone-pip data
    """
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.crossval,valfrac=stack.valfrac)
    
def parm50(stack):
    """
    Specifically for batch293 tone-pip data
    """
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=50,avg_resp=True)
    stack.append(nm.crossval,valfrac=stack.valfrac)

def fb24ch200(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=200,avg_resp=True)
    stack.append(nm.crossval,valfrac=stack.valfrac)
    
def fb24ch100(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,avg_resp=True) #Data not preprocessed to 100 Hz, internally converts
    stack.append(nm.crossval,valfrac=stack.valfrac)
    
def fb24ch100n(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,avg_resp=True) #Data not preprocessed to 100 Hz, internally converts
    stack.nests=20
    stack.append(nm.crossval,valfrac=stack.valfrac)
    
def fb18ch100(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.crossval,valfrac=stack.valfrac)
    
def fb18ch100u(stack):
    """
    keyword to load data and use without averaging trials (unaveraged), as would use for pupil data
    """
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.crossval,valfrac=stack.valfrac)
      
def fb18ch50(stack):
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=50,avg_resp=True)
    #stack.append(nm.crossval,valfrac=stack.valfrac)
    stack.append(nm.standard_est_val,valfrac=stack.valfrac)


def loadlocal(stack):
    """
    This keyword is just to load up a local file that is not yet on the BAPHY database.
    Right now just loads files from my computer --njs, June 27 2017
    """
    file='/Users/HAL-9000/Desktop/CompNeuro/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'_nat_export.mat'
    #file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.crossval,valfrac=0.05,use_trials=True)
    #stack.append(nm.standard_est_val, valfrac=0.05)
    stack.append(nm.pupil_model,tile_data=True)
    


#Est/val now incorporated into most loader keywords, but these still work if 
#crossval is not included in the loader keyword for some reason

def ev(stack):
    """
    DEPRECATED, not sure this actually works anymore. Fucntionality moved to 
    crossval module anyway though
    """
    stack.append(nm.standard_est_val, valfrac=0.05)
    
def xval10(stack):
    #stack.nests=10
    stack.append(nm.crossval,valfrac=0.1)
    
def xval05(stack):
    #stack.nests=20
    stack.append(nm.crossval,valfrac=0.05)
    


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
    stack.fitter.tol=0.0001
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
def nonlin_mini_fit(stack):
    # mini fit
    stack.append(nm.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.fitter=nf.basic_min(stack)
    stack.fitter.tol=0.00001
    #stack.fitter=nf.coordinate_descent(stack)
    #stack.fitter.tol=0.001
    fitidx=nu.find_modules(stack,'nonlinearity')
    stack.fitter.fit_modules=fitidx
    
    stack.fitter.do_fit()
    stack.popmodule()
def dlog(stack):
    stack.append(nm.nonlinearity,nltype='dlog',fit_fields=['phi'],phi=[1])
    
def exp(stack):
    stack.append(nm.nonlinearity,nltype='exp',fit_fields=['phi'],phi=[1,1])

def dexp(stack):
    stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['phi'],phi=[1,.01,.001,0]) 
    #choose phi s.t. dexp starts as almost a straight line 
    nonlin_mini_fit(stack)
    
def poly01(stack):
    stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1])
    
def poly02(stack):
    stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1,0])
    
def poly03(stack):
    stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1,0,0])

def tanhsig(stack):
    stack.append(nm.nonlinearity,nltype='tanh',fit_fields=['phi'],phi=[1,1,0])


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
    
def polypupgain02(stack): #2nd degree polynomial gain fn
    stack.append(nm.state_gain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,1])
    
def exppupgain(stack):
    stack.append(nm.state_gain,gain_type='exppupgain',fit_fields=['theta'],theta=[0,1,0,0])

def logpupgain(stack):
    stack.append(nm.state_gain,gain_type='logpupgain',fit_fields=['theta'],theta=[0,1,0,1])

def powergain02(stack): #This is equivalent ot what Zach is using
    stack.append(nm.state_gain,gain_type='powerpupgain',fit_fields=['theta'],theta=[0,1,0,0],order=2)
    
def butterworth01(stack):
    stack.append(nm.state_gain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=1)
    
def butterworth02(stack):
    stack.append(nm.state_gain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=2)
    
def butterworth03(stack):
    stack.append(nm.state_gain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=3)
    
def butterworth04(stack):
    stack.append(nm.state_gain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=4)


# fitter keywords
###############################################################################

def fit00(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
        
    stack.evaluate(2)

    stack.fitter=nf.basic_min(stack)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit01(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
        
    #stack.evaluate(2)

    stack.fitter=nf.basic_min(stack)
    stack.fitter.tol=0.00000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit02(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
        
    stack.evaluate(2)

    stack.fitter=nf.basic_min(stack,routine='SLSQP')
    stack.fitter.tol=0.000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit00h1(stack):
    hubidx=nu.find_modules(stack,'pseudo_huber_error')
    if not hubidx:
        stack.append(nm.pseudo_huber_error,b=1.0)
        stack.error=stack.modules[-1].error
    stack.evaluate(2)
    
    stack.fitter=nf.basic_min(stack)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    create_parmlist(stack)
    stack.popmodule()
    stack.append(nm.mean_square_error)
    
def fitannl00(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
    
    stack.evaluate(2)
    
    stack.fitter=nf.anneal_min(stack,anneal_iter=50,stop=5,up_int=10,bounds=None)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
    
def fitannl01(stack):
    mseidx=nu.find_modules(stack,'mean_square_error')
    if not mseidx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.mean_square_error)
        
        # set error (for minimization) for this stack to be output of last module
        stack.error=stack.modules[-1].error
    
    stack.evaluate(2)
    
    stack.fitter=nf.anneal_min(stack,anneal_iter=100,stop=10,up_int=5,bounds=None)
    stack.fitter.tol=0.000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fititer00(stack):
    
    stack.append(nm.mean_square_error,shrink=0.5)
    stack.error=stack.modules[-1].error
    
    stack.fitter=nf.fit_iteratively(stack,max_iter=5)
    #stack.fitter.sub_fitter=nf.basic_min(stack)
    stack.fitter.sub_fitter=nf.coordinate_descent(stack,tol=0.001,maxit=10,verbose=False)
    stack.fitter.sub_fitter.step_init=0.05
    
    stack.fitter.do_fit()
    create_parmlist(stack)


# etc etc for other keywords
###############################################################################

def perfectpupil100(stack):
    """keyword to fit pupil gain using "perfect" model generated by pupil_model module.
    The idea here is not to fit a model to the data, but merely to see the effect of a 
    a linear pupil gain function on the "perfect" model generated by averaging the
    rasters of each trial for a given stimulus. This keyword loads up the data
    and generates the model. It should be used with pupgain and a fitter keyword.
    """
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=100,avg_resp=False)
    #stack.nests=20
    stack.append(nm.crossval,valfrac=stack.valfrac)
    #stack.append(nm.standard_est_val, valfrac=0.05)
    stack.append(nm.pupil_model)
    
    
def perfectpupil50(stack):
    """keyword to fit pupil gain using "perfect" model generated by pupil_model module.
    The idea here is not to fit a model to the data, but merely to see the effect of a 
    a linear pupil gain function on the "perfect" model generated by averaging the
    rasters of each trial for a given stimulus. This keyword loads up the data
    and generates the model. It should be used with pupgain and a fitter keyword.
    """
    file=baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.crossval,valfrac=stack.valfrac)
    #stack.nests=20
    #stack.append(nm.standard_est_val, valfrac=0.05)
    stack.append(nm.pupil_model)
 
    
# Nested Crossval
###############################################################################

def nested20(stack):
    """
    Keyword for 20-fold nested crossvalidation. Uses 5% validation chunks.
    """
    stack.nests=20
    stack.valfrac=0.05
    nest_helper(stack)
        
def nested10(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    """
    stack.nests=10
    stack.valfrac=0.1
    nest_helper(stack)
    
    
# Helper/Support Functions
###############################################################################

def create_parmlist(stack):
    """
    Helper function that assigns all fitted parameters for a model to a single (n,)
    phi vector and accociates it to the stack.parm_fits object
    """
    phi=[] 
    for idx,m in enumerate(stack.modules):
        this_phi=m.parms2phi()
        if this_phi.size:
            if stack.cv_counter==0:
                stack.fitted_modules.append(idx)
            phi.append(this_phi)
    phi=np.concatenate(phi)
    stack.parm_fits.append(phi)

def nest_helper(stack):
    """
    Helper function for implementing nested crossvalidation. Essentially sets up
    a loop with the estimation part of fit_single_model inside. 
    """
    stack.cond=False
    while stack.cond is False:
        print('iter loop='+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in range(0,len(stack.keywords)-1):
            f = globals()[stack.keywords[k]]
            f(stack)
        
        stack.cv_counter+=1




