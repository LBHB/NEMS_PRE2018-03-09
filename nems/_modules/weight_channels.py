#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:17:11 2017

@author: shofer
"""
from lib.nems_modules import nems_module
import numpy as np
import lib.nems_utils as nu

class weight_channels(nems_module):
    """
    weight_channels - apply a weighting matrix across a variable in the data
    stream. Used to provide spectral filters, directly imported from NARF.
    a helper function parm_fun can be defined to parameterized the weighting
    matrix. but by default the weights are each independent
    """
    name='weight_channels'
    user_editable_fields=['output_name','num_dims','coefs','baseline','phi','parm_fun']
    plot_fns=[nu.plot_strf,nu.plot_spectrogram]
    coefs=None
    baseline=np.zeros([1,1])
    num_chans=1
    parm_fun=None
    
    def my_init(self, num_dims=0, num_chans=1, baseline=np.zeros([1,1]), 
                fit_fields=['coefs'], parm_fun=None, phi=np.zeros([1,1])):
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0]['stim'].shape[0]
        self.num_dims=num_dims
        self.num_chans=num_chans
        self.baseline=baseline
        self.fit_fields=fit_fields
        if parm_fun:
            self.parm_fun=parm_fun
            self.coefs=parm_fun(phi)
        else:
            #self.coefs=np.ones([num_chans,num_dims])/num_dims/100
            self.coefs=np.random.normal(1,0.1,[num_chans,num_dims])/num_dims
        self.phi=phi
        
    def my_eval(self,X):
        #if not self.d_out:
        #    # only allocate memory once, the first time evaling. rish is that output_name could change
        if self.parm_fun:
            self.coefs=self.parm_fun(self.phi)
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        X=np.matmul(self.coefs,X)
        s=list(s)
        s[0]=self.num_chans
        Y=np.reshape(X,s)
        return Y