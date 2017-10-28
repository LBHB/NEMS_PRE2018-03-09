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
    
    def my_init(self,nltype='dlog',fit_fields=['phi'],phi=[1]):
        """
        nltype: type of nonlinearity
        fit_fields: name of fitted parameters
        phi: initial values for fitted parameters
        """
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.fit_fields=fit_fields
        self.nltype=nltype
        self.phi=np.array([phi])
        if nltype=='dlog':
            self.do_plot=self.plot_fns[2]
        else:
            self.do_plot=self.plot_fns[1]
        
    def dlog_fn(self,X):
        #threshold input so that minimum of X is 1 and min output is 0
        s_indices= (X+self.phi[0,0])<=1
        X[s_indices]=1-self.phi[0,0]
        Y=np.log(X+self.phi[0,0])
        return(Y)
    def exp_fn(self,X):
        Y=np.exp(self.phi[0,0]*(X-self.phi[0,1]))
        return(Y)
    def dexp_fn(self,X):
        Y=self.phi[0,0]-self.phi[0,1]*np.exp(-np.exp(self.phi[0,2]*(X-self.phi[0,3])))
        return(Y)
    def poly_fn(self,X):
        deg=self.phi.shape[1]
        Y=0
        for i in range(0,deg):
            Y+=self.phi[0,i]*np.power(X,i)
        return(Y)
    def tanh_fn(self,X):
        Y=self.phi[0,0]*np.tanh(self.phi[0,1]*X-self.phi[0,2])+self.phi[0,0]
        return(Y)
    def logsig_fn(self,X):
        # from Rabinowitz et al 2011
        a=self.phi[0,0]
        b=self.phi[0,1]
        c=self.phi[0,2]
        d=self.phi[0,3]
        Y=a+b/(1+np.exp(-(X-c)/d))
        return(Y)
        
    def my_eval(self,X):
        Z=getattr(self,self.nltype+'_fn')(X)
        return(Z)
        
