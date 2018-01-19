#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules that apply a non-pupil dependent nonlinearity to the data

Created on Fri Aug  4 13:39:49 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

import numpy as np


class gain(nems_module):
    """
    nonlinearity - apply a static nonlinearity to the data. This modules uses helper
    functions and a generic fit field ('phi') to specify the type of nonlinearity
    applied. Look at nems/keywords to see how to apply different nonlinearities, as
    each kind has a specific nltype and phi.

    @author: shofer
    """
    # Added helper functions and removed look up table --njs June 29 2017
    name = 'nonlin.gain'
    plot_fns = [nems.utilities.plot.pre_post_psth,
                nems.utilities.plot.io_scatter_smooth, nems.utilities.plot.plot_spectrogram]
    user_editable_fields = ['input_name',
                            'output_name', 'fit_fields', 'nltype', 'phi']
    phi = np.array([1])
    nltype = 'dlog'

    def my_init(self, nltype='dlog', fit_fields=['phi'], phi=None):
        """
        nltype: type of nonlinearity
        fit_fields: name of fitted parameters
        phi: initial values for fitted parameters expects 2d numpy matrix
        """
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.fit_fields = fit_fields
        self.nltype = nltype
        
        if nltype == 'dlog':
            self.do_plot = self.plot_fns[2]
        else:
            self.do_plot = self.plot_fns[1]
            
        if phi is None:
            # default depends on the nl type
            if nltype == 'dexp':
                phi = np.array([[0,1,1,1]])
            elif nltype == 'dlog':
                phi = np.array([[-2]])
            else:
                phi = np.array([[1]])
        
        self.phi=phi
        
    def dlog_fn(self, X):
        # threshold input so that minimum of X is 1 and min output is 0
        offset = self.phi[0,0]
           
        # soften effects of more extreme offsets
        if offset>4:
            adjoffset=4+(offset-4)/50
        elif offset<-4:
            adjoffset=-4+(offset+4)/50
        else:
            adjoffset=offset
        
        d = 10.0**adjoffset 
        
        # Offset from zero
        if self.phi.shape[0] > 1:
            zeroer = self.phi[1,:]
        else:
            zeroer = 0
        
        # Zero below threshold
        if self.phi.shape[0] > 2:
            zbt = self.phi[2,:]           
        else:
            zbt = 0
        
        X[X<zbt] = zbt
        X = X-zbt
        
        Y = np.log((X + d)/d) + zeroer
        
        #s_indices = (X + self.phi[0, 0]) <= 1
        #X[s_indices] = 1 - self.phi[0, 0]
        #Y = np.log(X + self.phi[0, 0])
        return Y

    def exp_fn(self, X):
        Y = np.exp(self.phi[0, 0] * (X - self.phi[0, 1]))
        return(Y)

    def dexp_fn(self, X):
        Y = self.phi[:, 0:1] + self.phi[:, 1:2] * \
            np.exp(-np.exp(-self.phi[:, 2:3] * (X - self.phi[:, 3:4])))
        return(Y)

    def poly_fn(self, X):
        deg = self.phi.shape[1]
        Y = 0
        for i in range(0, deg):
            Y += self.phi[0, i] * np.power(X, i)
        return(Y)

    def tanh_fn(self, X):
        Y = self.phi[0, 0] * np.tanh(self.phi[0, 1]
                                     * X - self.phi[0, 2]) + self.phi[0, 0]
        return(Y)

    def logsig_fn(self, X):
        # from Rabinowitz et al 2011
        a = self.phi[:, 0:1]
        b = self.phi[:, 1:2]
        c = self.phi[:, 2:3]
        d = self.phi[:, 3:4]
        Y = a + b / (1 + np.exp(-(X - c) / d))
        return(Y)

    def my_eval(self, X):
        Z = getattr(self, self.nltype + '_fn')(X)
        return(Z)

    def evaluate(self):
        del self.d_out[:]
        # create a copy of each input variable
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            
        X=self.unpack_data(self.input_name,est=True)
        Z = self.my_eval(X)
        self.pack_data(Z,self.output_name,est=True)
        
        if self.parent_stack.valmode:
            X=self.unpack_data(self.input_name,est=False)
            Z = self.my_eval(X)
            self.pack_data(Z,self.output_name,est=False)
            


