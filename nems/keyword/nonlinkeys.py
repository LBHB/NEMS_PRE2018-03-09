#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinearity keywords

Created on Fri Aug 11 11:08:08 2017

@author: shofer
"""
import nems.modules as nm
from nems.utilities.utils import mini_fit

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
    mini_fit(stack,mods=['nonlin.gain'])
    
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
    mini_fit(stack,mods=['nonlin.gain'])
    
def logsig(stack):
    phi=[0,1,0,1]
    stack.append(nm.nonlin.gain,nltype='logsig',fit_fields=['phi'],phi=phi) 
    mini_fit(stack,mods=['nonlin.gain'])
    
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
    mini_fit(stack,mods=['nonlin.gain'])