#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitter keywords

Created on Fri Aug 11 11:14:52 2017

@author: shofer
"""

import nems.modules as nm
import nems.fitters
#import nems.tensorflow_fitters as ntf
from nems.utilities.utils import create_parmlist


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

    stack.fitter=nems.fitters.fitters.basic_min(stack)
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

    stack.fitter=nems.fitters.fitters.basic_min(stack)
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

    stack.fitter=nems.fitters.fitters.basic_min(stack,routine='SLSQP')
    stack.fitter.tolerance=0.000001
    stack.fitter.do_fit()
    create_parmlist(stack)

def fit03(stack):
    """
    Fits the model parameters using a mean squared error loss function with 
    the L-BFGS-B algorithm, to a cost function tolerance of 10^-7.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter=nems.fitters.fitters.basic_min(stack)
    stack.fitter.tolerance=0.0000001
    stack.fitter.do_fit()
    create_parmlist(stack)
    
def fit04(stack):
    """
    Fits the model parameters using a Poisson LL loss function with 
    the L-BFGS-B algorithm, to a cost function tolerance of 10^-7.
    
    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.likelihood_poisson)
    stack.error=stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter=nems.fitters.fitters.basic_min(stack)
    stack.fitter.tolerance=0.0000001
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
    
    stack.fitter=nems.fitters.fitters.basic_min(stack)
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
    
    stack.fitter=nems.fitters.fitters.anneal_min(stack,anneal_iter=50,stop=5,up_int=10,bounds=None)
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
    
    stack.fitter=nems.fitters.fitters.anneal_min(stack,anneal_iter=100,stop=10,up_int=5,bounds=None)
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
    
    stack.fitter=nems.fitters.fitters.fit_iteratively(stack,max_iter=5)
    #stack.fitter.sub_fitter=nems.fitters.fitters.basic_min(stack)
    stack.fitter.sub_fitter=nems.fitters.fitters.coordinate_descent(stack,tolerance=0.001,maxit=10,verbose=False)
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
    
    stack.fitter=nems.fitters.fitters.fit_iteratively(stack,max_iter=5)
    stack.fitter.sub_fitter=nems.fitters.fitters.basic_min(stack)
    
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
