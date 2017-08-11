#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import nems.modules as nm
import nems.fitters as nf
#import nems.tensorflow_fitters as ntf
import nems.utilities as ut
import numpy as np
import nems.modules.user_def as ud


# loader keywords
def parm100(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def parm50(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def parm50a(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Averages the response 
    to each stimulus over its respective raster, and applies a 5% 
    estimation/validation split if the est/val datasets are not specified in 
    the file.
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)

def fb24ch200(stack):
    """
    Loads a 24 channel, 200 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=200,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def fb24ch100(stack):
    """
    Loads a 24 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True) #Data not preprocessed to 100 Hz, internally converts
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def fb18ch100(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def fb18ch100u(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Does not average over response rasters, instead treating each trial as a separate
    stimulus. Applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
      
def fb18ch50(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch,
    then downsamples to 50 Hz.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    #stack.append(nm.crossval,valfrac=stack.valfrac)
    stack.append(nm.est_val.standard_est_val,valfrac=stack.valfrac)

def loadlocal(stack):
    """
    This keyword is just to load up a local file that is not yet on the BAPHY database.
    Right now just loads files from my computer --njs, June 27 2017
    """
    file='/Users/HAL-9000/Desktop/CompNeuro/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'_b'+str(stack.meta['batch'])+'_ozgf_c18_fs100.mat'
    #file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval)
    
def jitterload(stack):
    """
    Loads a 2-channel, 1000 Hz .mat file containing data for Mateo's jitter tone
    project, and downsamples to 500 Hz. Applies a 5% estimation/validation split.
    """
    filepath='/auto/users/shofer/data/batch296mateo/'+str(stack.meta['cellid'])+'_b'+str(stack.meta['batch'])+'_envelope_fs1000.mat'
    print("Initializing load_mat with file {0}".format(filepath))
    stack.append(ud.load_baphy_ssa.load_baphy_ssa,file=filepath,fs=500)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)


#Est/val now incorporated into most loader keywords, but these still work if 
#crossval is not included in the loader keyword for some reason

def ev(stack):
    """
    Breaks the data into estimation and validation datasets based on the number
    of trials of each stimulus.
    """
    stack.append(nm.est_val.standard, valfrac=0.05)
    
def xval10(stack):
    """
    Breaks the data into estimation and validation datasets by placing 90% of the
    trials/stimuli into the estimation set and 10% into the validation set.
    """
    stack.append(nm.est_val.crossval,valfrac=0.1)
    
def xval05(stack):
    """
    Breaks the data into estimation and validation datasets by placing 95% of the
    trials/stimuli into the estimation set and 5% into the validation set.
    """
    stack.append(nm.est_val.crossval,valfrac=0.05)
    


# weight channels keywords
###############################################################################

def wc01(stack):
    """
    Applies a 1 channel spectral filter matrix to the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=1)

def wc02(stack):
    """
    Applies a 2 channel spectral filter matrix to the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=2)

def wc03(stack):
    """
    Applies a 3 channel spectral filter matrix to the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=3)

def wc04(stack):
    """
    Applies a 4 channel spectral filter matrix to a the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=4)


# fir filter keywords
###############################################################################

def fir_mini_fit(stack):
    """
    Helper function that fits weight channel and fir coefficients prior to fitting 
    all the model coefficients. This is often helpful, as it gets the model in the
    right ballpark before fitting other model parameters, especially when nonlinearities
    are included in the model.
    
    This function is not appended directly to the stack, but instead is included
    in fir keywords (see help for fir10, fir15)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tolerance=0.0001
    try:
        fitidx=ut.utils.find_modules(stack,'filters.weight_channels') + ut.utils.find_modules(stack,'filters.fir')
    except:
        fitidx=ut.utils.find_modules(stack,'fir')
    stack.fitter.fit_modules=fitidx
    
    stack.fitter.do_fit()
    stack.popmodule()
    
def fir10(stack):
    """
    Appends a 10 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword initializes the FIR coefficients to 0, and performs a fit on the 
    FIR coefficients (and weight channel coefficients, if a weight channel matrix
    is included in the model).
    """
    stack.append(nm.filters.fir,num_coefs=10)
    fir_mini_fit(stack)
    
def fir10r(stack):
    """
    Appends a 10 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword draws the intial FIR coeffcients from a normal distribution about 0 
    with a standard distribution of 0.0025, and performs a fit on the FIR coefficients 
    (and weight channel coefficients, if a weight channel matrix is included in the model).
    """
    stack.append(nm.filters.fir,num_coefs=10,random=True)
    fir_mini_fit(stack)
    
def fir15(stack):
    """
    Appends a 15 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword initializes the FIR coefficients to 0, and performs a fit on the 
    FIR coefficients (and weight channel coefficients, if a weight channel matrix
    is included in the model).
    """
    stack.append(nm.filters.fir,num_coefs=15)
    fir_mini_fit(stack)

def stp1pc(stack):
    #stack.append(nm.aux.normalize)
    #stack.append(nm.filters.stp,num_channels=1,fit_fields=[])
    stack.append(nm.filters.stp,num_channels=1)
    stack.modules[-1].u[:]=0.05
    
def stp2pc(stack):
    stack.append(nm.filters.stp,num_channels=2)
    stack.modules[-1].u[:,0]=0.01
    stack.modules[-1].u[:,1]=0.1
    
    

# static NL keywords
###############################################################################
def nonlin_mini_fit(stack):
    """
    Helper function that fits nonlinearity coefficients prior to fitting 
    all the model coefficients. This is often helpful, as just fitting the
    nonlinearities tends to reduce the tendency of the nonlinearities to send 
    the predicted response to a constant DC offset.
    
    This function is not appended directly to the stack, but instead is included
    in certain nonlinearity keywords (see help for dexp, tanhsig)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tolerance=0.0001
    fitidx=ut.utils.find_modules(stack,'nonlin.gain')
    stack.fitter.fit_modules=fitidx
    
    stack.fitter.do_fit()
    stack.popmodule()
    
def dlog(stack):
    """
    Applies a natural logarithm entry-by-entry to the datastream: 
        y = log(x+v1)
    where x is the input matrix and v1 is a fitted parameter applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain,nltype='dlog',fit_fields=['phi'],phi=[1])
    #stack.append(nm.normalize)
    
def exp(stack):
    """
    Applies an exponential function entry-by-entry to the datastream:
        y = exp(v1*(x+v2))
    where x is the input matrix and v1, v2 are fitted parameters applied to each
    matrix entry (the same across all entries)
    
    Performs a fit on the nonlinearity parameters, as well.
    """
    stack.append(nm.nonlin.gain,nltype='exp',fit_fields=['phi'],phi=[1,1])
    nonlin_mini_fit(stack)

def dexp(stack):
    """
    Applies a double-exponential function entry-by-entry to the datastream:
        y = v1 - v2*exp[-exp{v3*(x-v4)}]
    where x is the input matrix and v1,v2,v3,v4 are fitted parameters applied to each
    matrix entry (the same across all entries)
    
    Performs a fit on the nonlinearity parameters, as well.
    """
    stack.append(nm.nonlin.gain,nltype='dexp',fit_fields=['phi'],phi=[1,.01,.001,0]) 
    #choose phi s.t. dexp starts as almost a straight line 
    nonlin_mini_fit(stack)
    
def logsig(stack):
    phi=[0,1,0,1]
    stack.append(nm.nonlin.gain,nltype='logsig',fit_fields=['phi'],phi=phi) 
    #choose phi s.t. dexp starts as almost a straight line 
    nonlin_mini_fit(stack)
    
def poly01(stack):
    """
    Applies a polynomial function entry-by-entry to the datastream:
        y = v1 + v2*x
    where x is the input matrix and v1,v2 are fitted parameters applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain,nltype='poly',fit_fields=['phi'],phi=[0,1])
    
def poly02(stack):
    """
    Applies a polynomial function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*(x^2)
    where x is the input matrix and v1,v2,v3 are fitted parameters applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain,nltype='poly',fit_fields=['phi'],phi=[0,1,0])
    
def poly03(stack):
    """
    Applies a polynomial function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*(x^2) + v4*(x^3)
    where x is the input matrix and v1,v2,v3,v4 are fitted parameters applied to 
    each matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain,nltype='poly',fit_fields=['phi'],phi=[0,1,0,0])

def tanhsig(stack):
    """
    Applies a tanh sigmoid function(commonly used as a neural net activation function) 
    entry-by-entry to the datastream:
        y = v1*tanh(v2*x - v3) + v1
    where x is the input matrix and v1,v2,v3 are fitted parameters applied to 
    each matrix entry (the same across all entries)
    
    Performs a fit on the nonlinearity parameters, as well.
    """
    stack.append(nm.nonlin.gain,nltype='tanh',fit_fields=['phi'],phi=[1,1,0])
    nonlin_mini_fit(stack)


# state variable keyowrds
###############################################################################
def pupil_mini_fit(stack):
    """
    Helper function that fits nonlinearity coefficients prior to fitting 
    all the model coefficients. This is often helpful, as just fitting the
    nonlinearities tends to reduce the tendency of the nonlinearities to send 
    the predicted response to a constant DC offset.
    
    This function is not appended directly to the stack, but instead is included
    in certain nonlinearity keywords (see help for dexp, tanhsig)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tolerance=0.00001
    fitidx=ut.utils.find_modules(stack,'pupil.pupgain')
    stack.fitter.fit_modules=fitidx
    
    stack.fitter.do_fit()
    stack.popmodule()
    
def nopupgain(stack):
    """
    Applies a DC gain function entry-by-entry to the datastream:
        y = v1 + v2*x
    where x is the input matrix and v1,v2 are fitted parameters applied to 
    each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='nopupgain',fit_fields=['theta'],theta=[0,1])
    pupil_mini_fit(stack)
    
def pupgainctl(stack):
    """
    Applies a DC gain function entry-by-entry to the datastream:
        y = v1 + v2*x + <randomly shuffled pupil dc-gain>
    where x is the input matrix and v1,v2 are fitted parameters applied to 
    each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='linpupgainctl',fit_fields=['theta'],theta=[0,1,0,0])
    pupil_mini_fit(stack)
    
def pupgain(stack):
    """
    Applies a linear pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*p + v4*x*p
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2 
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0])
    pupil_mini_fit(stack)

def polypupgain04(stack):#4th degree polynomial gain fn
    """
    Applies a poynomial pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*x*p + v4*x*(p^2) + v5*x*(p^3) + v6*x*(p^4)
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2,v3,v4,v5,v6 
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,0,0,1])
    pupil_mini_fit(stack)
    
def polypupgain03(stack): #3rd degree polynomial gain fn
    """
    Applies a poynomial pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*x*p + v4*x*(p^2) + v5*x*(p^3)
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2,v3,v4,v5 
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,0,1])
    pupil_mini_fit(stack)
    
def polypupgain02(stack): #2nd degree polynomial gain fn
    """
    Applies a poynomial pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*x*p + v4*x*(p^2) 
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2,v3,v4
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='polypupgain',fit_fields=['theta'],theta=[0,0,0,1])
    pupil_mini_fit(stack)
    
def exppupgain(stack):
    """
    Applies an exponential pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x*exp(v3*p-v4)
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2,v3,v4
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='exppupgain',fit_fields=['theta'],theta=[0,1,0,0])
    pupil_mini_fit(stack)

def logpupgain(stack):
    """
    Applies a logarithmic pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x*log(v3*p-v4)
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2,v3,v4
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='logpupgain',fit_fields=['theta'],theta=[0,1,0,1])
    pupil_mini_fit(stack)
    
def powergain02(stack): #This is equivalent ot what Zach is using
    """
    Applies a poynomial pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*(p^2) + v4*x*(p^2) 
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2,v3,v4
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    stack.append(nm.pupil.pupgain,gain_type='powerpupgain',fit_fields=['theta'],theta=[0,1,0,0],order=2)
    pupil_mini_fit(stack)
    
def butterworth01(stack):
    """
    Applies a 1st-order Butterworth high-pass filter entry-by-entry to the datastream,
    where the pupil diameter is used as the "frequency", and the -3dB is fit, along
    with a scalar gain term and a DC offset. 
    """
    stack.append(nm.pupil.pupgain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=1)
    pupil_mini_fit(stack)
    
def butterworth02(stack):
    """
    Applies a 2nd-order Butterworth high-pass filter entry-by-entry to the datastream,
    where the pupil diameter is used as the "frequency", and the -3dB is fit, along
    with a scalar gain term and a DC offset. 
    """
    stack.append(nm.pupil.pupgain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=2)
    pupil_mini_fit(stack)
    
def butterworth03(stack):
    """
    Applies a 3rd-order Butterworth high-pass filter entry-by-entry to the datastream,
    where the pupil diameter is used as the "frequency", and the -3dB is fit, along
    with a scalar gain term and a DC offset. 
    """
    stack.append(nm.pupil.pupgain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=3)
    pupil_mini_fit(stack)
    
def butterworth04(stack):
    """
    Applies a 4th-order Butterworth high-pass filter entry-by-entry to the datastream,
    where the pupil diameter is used as the "frequency", and the -3dB is fit, along
    with a scalar gain term and a DC offset. 
    """
    stack.append(nm.pupil.pupgain,gain_type='butterworthHP',fit_fields=['theta'],theta=[1,25,0],order=4)
    pupil_mini_fit(stack)

# fitter keywords
###############################################################################

def fit00(stack):
    """
    Fits the model parameters using a mean squared error loss function with 
    the L-BFGS-B algorithm, to a cost function tolerance of 0.001.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)  
    # set error (for minimization) for this stack to be output of last module
    stack.error=stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tolerance=0.001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit01(stack):
    """
    Fits the model parameters using a mean squared error loss function with 
    the L-BFGS-B algorithm, to a cost function tolerance of 10^-8.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tolerance=0.00000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit02(stack):
    """
    Fits the model parameters using a mean squared error loss function with 
    the SLSQP algorithm, to a cost function tolerance of 10^-6.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter=nf.fitters.basic_min(stack,routine='SLSQP')
    stack.fitter.tolerance=0.000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit00h1(stack):
    """
    Fits the model parameters using a pseudo-huber loss function with 
    the L-BFGS-B algorithm, to a cost function tolerance of 0.001.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.pseudo_huber_error,b=1.0)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)
    
    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    create_parmlist(stack)
    stack.popmodule()
    stack.append(nm.metrics.mean_square_error)
    
def fitannl00(stack):
    """
    Fits the model parameters using a simulated annealing fitting procedure.
    
    Each repetition of annealing is performed with a mean_square_error cost function 
    using the L-BFGS-B algorithm, to a tolerance of 0.001. 
    
    50 rounds of annealing are performed, with the step size updated dynamically
    every 10 rounds. The routine will stop if the minimum function value remains
    the same for 5 rounds of annealing.
    
    Note that this routine takes a long time.
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)
    
    stack.fitter=nf.fitters.anneal_min(stack,anneal_iter=50,stop=5,up_int=10,bounds=None)
    stack.fitter.tol=0.001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
    
def fitannl01(stack):
    """
    Fits the model parameters using a simulated annealing fitting procedure.
    
    Each repetition of annealing is performed with a mean_square_error cost function 
    using the L-BFGS-B algorithm, to a tolerance of 10^-6. 
    
    100 rounds of annealing are performed, with the step size updated dynamically
    every 5 rounds. The routine will stop if the minimum function value remains
    the same for 10 rounds of annealing.
    
    Note that this routine takes a very long time.
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)
    
    stack.fitter=nf.fitters.anneal_min(stack,anneal_iter=100,stop=10,up_int=5,bounds=None)
    stack.fitter.tol=0.000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fititer00(stack):
    """
    Fits the model parameters using a mean-squared-error loss function with 
    a coordinate descent algorithm. However, rather than fitting all model 
    parameters at once, it only fits the parameters for one model at a time.
    The routine fits each module to a tolerance of 0.001, than halves the tolerance
    and repeats up to 9 more times.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error,shrink=0.5)
    stack.error=stack.modules[-1].error
    
    stack.fitter=nf.fitters.fit_iteratively(stack,max_iter=5)
    #stack.fitter.sub_fitter=nf.fitters.basic_min(stack)
    stack.fitter.sub_fitter=nf.fitters.coordinate_descent(stack,tolerance=0.001,maxit=10,verbose=False)
    stack.fitter.sub_fitter.step_init=0.05
    
    stack.fitter.do_fit()
    create_parmlist(stack)

def fititer01(stack):
    """
    Fits the model parameters using a mean-squared-error loss function with 
    a coordinate descent algorithm. However, rather than fitting all model 
    parameters at once, it only fits the parameters for one model at a time.
    The routine fits each module to a tolerance of 0.001, than halves the tolerance
    and repeats up to 9 more times.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error,shrink=0.5)
    stack.error=stack.modules[-1].error
    
    stack.fitter=nf.fitters.fit_iteratively(stack,max_iter=5)
    stack.fitter.sub_fitter=nf.fitters.basic_min(stack)
    
    stack.fitter.do_fit()
    create_parmlist(stack)

#def adadelta00(stack):
#    """
#    Very unoperational attempt at using tensorflow
#    """
#    stack.fitter=ntf.ADADELTA_min(stack)
#   stack.fitter.do_fit()
#    create_parmlist(stack)
#    stack.append(nm.mean_square_error)




# etc etc for other keywords
###############################################################################

def perfectpupil100(stack):
    """keyword to fit pupil gain using "perfect" model generated by pupil_model module.
    The idea here is not to fit a model to the data, but merely to see the effect of a 
    a linear pupil gain function on the "perfect" model generated by averaging the
    rasters of each trial for a given stimulus. This keyword loads up the data
    and generates the model. It should be used with pupgain and a fitter keyword.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    #stack.nests=20
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    #stack.append(nm.standard_est_val, valfrac=0.05)
    stack.append(nm.pupil.model)
    
    
def perfectpupil50(stack):
    """keyword to fit pupil gain using "perfect" model generated by pupil_model module.
    The idea here is not to fit a model to the data, but merely to see the effect of a 
    a linear pupil gain function on the "perfect" model generated by averaging the
    rasters of each trial for a given stimulus. This keyword loads up the data
    and generates the model. It should be used with pupgain and a fitter keyword.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    #stack.nests=20
    #stack.append(nm.standard_est_val, valfrac=0.05)
    stack.append(nm.pupil.model)
    
 
    
# Nested Crossval
###############################################################################

def nested20(stack):
    """
    Keyword for 20-fold nested crossvalidation. Uses 5% validation chunks. 
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=20
    stack.valfrac=0.05
    nest_helper(stack)
        
def nested10(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=10
    stack.valfrac=0.1
    nest_helper(stack)
    
def nested5(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=5
    stack.valfrac=0.2
    nest_helper(stack)
    
# DEMO KEYWORDS
###############################################################################

def simpledemo00(stack):
    """
    Keyword for the simple_demo DC gain module. Appends the simple_demo module
    with its default arguments.
    
    Applies a simple DC gain and offset to the input data:
        y = v1*x + v2
    where x is the input variable, and v1,v2 are fitted parameters. 
    
    """
    stack.append(ud.demo.simple_demo)
    
def simpledemo01(stack):
    """
    Keyword for the simple_demo DC gain module. Uses arguments different than the
    default.
    
    Applies a simple DC gain and offset to the input data with a 0 output threshold:
        y = v1*x + v2
    where x is the input variable, and v1,v2 are fitted parameters. 
    """
    stack.append(ud.demo.simple_demo,thresh=True)
    
def advdemo00(stack):
    stack.append(ud.demo.adv_demo)
    
    
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
        print('Nest #'+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in range(0,len(stack.keywords)-1):
            f = globals()[stack.keywords[k]]
            f(stack)
        
        stack.cv_counter+=1




