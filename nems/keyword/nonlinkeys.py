#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinearity keywords

Created on Fri Aug 11 11:08:08 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

from functools import partial

import numpy as np
import nems.modules as nm
from nems.utilities.utils import mini_fit

from .registry import keyword_registry


def dlog2(stack):
    """
    Applies a natural logarithm entry-by-entry to the datastream:
        y = log(x+v1)
    where x is the input matrix and v1 is a fitted parameter applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain, nltype='dlog', fit_fields=[], phi=[1])


def dlog(stack):
    """
    Applies a natural logarithm entry-by-entry to the datastream:
        y = log(x+v1)
    where x is the input matrix and v1 is a fitted parameter applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain, nltype='dlog', fit_fields=['phi'], phi=[-2])


def exp(stack):
    """
    Applies an exponential function entry-by-entry to the datastream:
        y = exp(v1*(x+v2))
    where x is the input matrix and v1, v2 are fitted parameters applied to each
    matrix entry (the same across all entries)

    Performs a fit on the nonlinearity parameters, as well.
    """
    stack.append(nm.nonlin.gain, nltype='exp', fit_fields=['phi'], phi=[1, 1])
    mini_fit(stack, mods=['nonlin.gain'])


def dexp(stack):
    """
    Applies a double-exponential function entry-by-entry to the datastream:
        y = v1 - v2*exp[-exp{v3*(x-v4)}]
    where x is the input matrix and v1,v2,v3,v4 are fitted parameters applied to each
    matrix entry (the same across all entries)

    Performs a fit on the nonlinearity parameters, as well.
    """
    
    # load the incoming pred/resp to guess at initial conditions
    resp = stack.modules[-1].unpack_data('resp', use_dout=True)
    pred = stack.modules[-1].unpack_data('pred', use_dout=True)
    if resp.shape[0]>pred.shape[0]:
        resp=resp[:pred.shape[0],:]
    keepidx = np.isfinite(resp) * np.isfinite(pred)
    resp = resp[keepidx]
    pred = pred[keepidx]

    # choose phi s.t. dexp starts as almost a straight line
    # phi=[max_out min_out slope mean_in]
    meanr = np.nanmean(resp)
    stdr = np.nanstd(resp)
    phi = [meanr + stdr * 4, stdr * 8, np.std(pred) / 10, np.mean(pred)]
    log.info(phi)
    stack.append(nm.nonlin.gain, nltype='dexp', fit_fields=['phi'], phi=phi)
    mini_fit(stack, mods=['nonlin.gain'])


def logsig(stack):
    #        a=self.phi[0,0]
    #        b=self.phi[0,1]
    #        c=self.phi[0,2]
    #        d=self.phi[0,3]
    #        Y=a+b/(1+np.exp(-(X-c)/d))
    resp = stack.modules[-1].unpack_data('resp', use_dout=True)
    pred = stack.modules[-1].unpack_data('pred', use_dout=True)
    meanr = np.mean(resp)
    stdr = np.std(resp)
    phi = [meanr - stdr * 3, stdr * 6, np.mean(pred), np.std(pred)]
    stack.append(nm.nonlin.gain, nltype='logsig', fit_fields=['phi'], phi=phi)
    mini_fit(stack, mods=['nonlin.gain'])


def poly01(stack):
    """
    Applies a polynomial function entry-by-entry to the datastream:
        y = v1 + v2*x
    where x is the input matrix and v1,v2 are fitted parameters applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain, nltype='poly', fit_fields=['phi'], phi=[0, 1])


def poly02(stack):
    """
    Applies a polynomial function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*(x^2)
    where x is the input matrix and v1,v2,v3 are fitted parameters applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain, nltype='poly',
                 fit_fields=['phi'], phi=[0, 1, 0])


def poly03(stack):
    """
    Applies a polynomial function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*(x^2) + v4*(x^3)
    where x is the input matrix and v1,v2,v3,v4 are fitted parameters applied to
    each matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain, nltype='poly',
                 fit_fields=['phi'], phi=[0, 1, 0, 0])


def tanhsig(stack):
    """
    Applies a tanh sigmoid function(commonly used as a neural net activation function)
    entry-by-entry to the datastream:
        y = v1*tanh(v2*x - v3) + v1
    where x is the input matrix and v1,v2,v3 are fitted parameters applied to
    each matrix entry (the same across all entries)

    Performs a fit on the nonlinearity parameters, as well.
    """
    stack.append(nm.nonlin.gain, nltype='tanh',
                 fit_fields=['phi'], phi=[1, 1, 0])
    mini_fit(stack, mods=['nonlin.gain'])


keyword_registry.update({
    'poly01': poly01,
    'poly02': poly02,
    'poly03': poly03,
    'tanhsig': tanhsig,
    'logsig': logsig,
    'exp': exp,
    'dexp': dexp,
    'dlog': dlog,
    'dlog2': dlog2,
})
