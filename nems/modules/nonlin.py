#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules that apply a non-pupil dependent nonlinearity to the data

Created on Fri Aug  4 13:39:49 2017

@author: shofer
"""
from nems.modules.base import nems_module
import nems.utilities.utils as nu

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
    name='nonlinearity'
    plot_fns=[nu.pre_post_psth,nu.io_scatter_smooth,nu.plot_spectrogram]
    user_editable_fields = ['nltype', 'fit_fields','phi']
    phi=np.array([1])
    
    def my_init(self,d_in=None,my_eval=None,nltype='dlog',fit_fields=['phi'],phi=[1],premodel=False):
        """
        
        """
        if premodel is True:
            self.do_plot=self.plot_fns[2]
        self.fit_fields=fit_fields
        self.nltype=nltype
        self.phi=np.array([phi])
        #setattr(self,nltype,phi0)
        if my_eval is None:
            #TODO: could do this more cleanly, see pupil.gain
            if nltype=='dlog':
                self.my_eval=self.dlog_fn
                self.plot_fns=[nu.plot_spectrogram]
                self.do_plot=self.plot_fns[0]
            elif nltype=='exp':
                self.my_eval=self.exp_fn
                self.do_plot=self.plot_fns[1]
            elif nltype=='dexp':
                self.my_eval=self.dexp_fn
                self.do_plot=self.plot_fns[1]
        else:
            self.my_eval=my_eval
            
        
    def dlog_fn(self,X):
        s_indices= X<=0
        X[s_indices]=0
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
        
    def my_eval(self,X):
        Z=getattr(self,self.nltype+'_fn')(X)
        return(Z)
        
