#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinearity keywords
"""
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
    stack.append(nm.nonlin.gain,nltype='dlog',fit_fields=[],phi=[1])


def dlog(stack):
    """
    Applies a natural logarithm entry-by-entry to the datastream:
        y = log(x+v1)
    where x is the input matrix and v1 is a fitted parameter applied to each
    matrix entry (the same across all entries)
    """
    stack.append(nm.nonlin.gain,nltype='dlog',fit_fields=['phi'],phi=[1])


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
    try:
        # TODO. This seems like something that should be pushed down into the
        # fitter class. Or, come up with a default approach?
        resp=stack.modules[-1].unpack_data('resp',use_dout=True)
        pred=stack.modules[-1].unpack_data('pred',use_dout=True)
        keepidx=np.isfinite(resp) * np.isfinite(pred)
        resp=resp[keepidx]
        pred=pred[keepidx]

        #choose phi s.t. dexp starts as almost a straight line
        meanr=np.nanmean(resp)
        stdr=np.nanstd(resp)
        phi=[meanr+stdr*4, stdr*8, np.std(pred)/10, np.mean(pred)]
    except AttributeError:
        phi = np.zeros(4)
    module = nm.nonlin.gain(nltype='dexp',fit_fields=['phi'],phi=phi)
    stack.append(module)

    # This is a hack. Again, this seems like it should be moved to the fitters.
    if not isinstance(stack, list):
        mini_fit(stack,mods=['nonlin.gain'])


def logsig(stack):
    resp=stack.modules[-1].unpack_data('resp',use_dout=True)
    pred=stack.modules[-1].unpack_data('pred',use_dout=True)
    meanr=np.mean(resp)
    stdr=np.std(resp)
    phi=[meanr-stdr*3,stdr*6,np.mean(pred),np.std(pred)]
    stack.append(nm.nonlin.gain,nltype='logsig',fit_fields=['phi'],phi=phi)
    mini_fit(stack,mods=['nonlin.gain'])


def poly(stack, degree=1):
    """
    Applies a polynomial function of the specified degree.
    """
    phi = np.zeros(degree+1)
    phi[1] = 1
    module = nm.nonlin.gain(nltype='poly', fit_fields=['phi'], phi=phi)
    stack.append(module)


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


def lognn(stack):
    module = nm.nonlin.NormalizeChannels(force_positive=True)
    stack.append(module)
    module = nm.nonlin.gain(nltype='log', fit_fields=['phi'], phi=np.zeros(3))
    stack.append(module)


keyword_registry.update({
    'poly01': partial(poly, degree=1),
    'poly02': partial(poly, degree=2),
    'poly03': partial(poly, degree=3),
    'tanhsig': tanhsig,
    'poly': poly,
    'logsig': logsig,
    'exp': exp,
    'dexp': dexp,
    'dlog': dlog,
    'dlog2': dlog2,
    'lognn': lognn,
})
