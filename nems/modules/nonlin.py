#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules that apply a non-pupil dependent nonlinearity to the data

Created on Fri Aug  4 13:39:49 2017

@author: shofer
"""
from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

import numpy as np


class NormalizeChannels(nems_module):

    def my_init(self, force_positive=False):
        self.force_positvie = force_positive

    def my_eval(self, x):
        return x


def dexp(phi, x):
    '''
    Defines a double-exponential sigmoid that's naturally asymmetric.
    '''
    base, peak = min(phi[:2]), max(phi[:2])
    lrshift = phi[2]
    kappa = phi[3]
    return base + peak * np.exp(-np.exp(-kappa*(x-lrshift)))


def log(phi, x):
    '''
    Parmeters
    ---------
    phi : array-like (length 3)
        First term is the curvature of the logarithm (i.e., the logarithm base),
        second term is the zero offset (i.e,, baseline rate) and third term
        specifies an input threshold. Inputs below this threshold are ignored.
    x: array-like
        Input to transform
    '''
    #base, offset, threshold = phi
    base = phi
    if base > 4:
        base = 4 + (offset-4)/50;
    elif base < -4:
        base = -4 + (offset+4)/50;

    d = 10**base

    #mask = x < threshold
    #x[mask] = threshold
    #x -= threshold
    #return np.log((x + d)/d) + offset
    return np.log((x + d)/d)


nl_functions = {
    'dexp': dexp,
    'log': log,
}


class gain(nems_module):
    """
    nonlinearity - apply a static nonlinearity to the data. This modules uses helper
    functions and a generic fit field ('phi') to specify the type of nonlinearity
    applied. Look at nems/keywords to see how to apply different nonlinearities, as
    each kind has a specific nltype and phi.

    @author: shofer
    """
    #Added helper functions and removed look up table --njs June 29 2017
    name='nonlin.gain'
    plot_fns=[nems.utilities.plot.pre_post_psth,nems.utilities.plot.io_scatter_smooth,nems.utilities.plot.plot_spectrogram]
    user_editable_fields = ['input_name','output_name','fit_fields','nltype','phi']
    phi=np.array([1])
    nltype='dlog'

    def my_init(self, nltype='dlog', fit_fields=['phi'], phi=[1]):
        """
        nltype: type of nonlinearity
        fit_fields: name of fitted parameters
        phi: initial values for fitted parameters
        """
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.fit_fields=fit_fields
        self.nltype=nltype
        self.phi=np.array(phi)
        if nltype=='dlog':
            self.do_plot=self.plot_fns[2]
        else:
            self.do_plot=self.plot_fns[1]

    #def dlog_fn(self, phi, x):
    #    #threshold input so that minimum of X is 1 and min output is 0
    #    s_indices= (X+self.phi[0,0])<=1
    #    X[s_indices]=1-self.phi[0,0]
    #    Y=np.log(X+self.phi[0,0])
    #    return(Y)
    #def exp_fn(self,X):
    #    a, b = pii
    #    Y=np.exp(self.phi[0,0]*(X-self.phi[0,1]))
    #    return(Y)
    def dexp_fn(self,X):
        Y=self.phi[0,0]-self.phi[0,1]*np.exp(-np.exp(self.phi[0,2]*(X-self.phi[0,3])))
        return(Y)
    #def poly_fn(self,X):
    #    deg=self.phi.shape[1]
    #    Y=0
    #    for i in range(0,deg):
    #        Y+=self.phi[0,i]*np.power(X,i)
    #    return(Y)

    #def tanh_fn(self,X):
    #    Y=self.phi[0,0]*np.tanh(self.phi[0,1]*X-self.phi[0,2])+self.phi[0,0]
    #    return(Y)

    #def logsig_fn(self, phi, X):
    #    # from Rabinowitz et al 2011
    #    a, b, c, d = phi
    #    return a+b/(1+np.exp(-(x-c)/d))

    def get_phi(self):
        print(self.phi)
        print(self.phi)
        print(self.phi)
        return self.phi

    def my_eval(self, x):
        f = nl_functions[self.nltype]
        phi = self.get_phi()
        return f(phi, x)

