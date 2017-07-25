#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:03:10 2017

@author: shofer
"""
from lib.nems_modules import nems_module
import lib.nems_utils as nu
import numpy as np




class nonlinearity(nems_module): 
    """
    nonlinearity - apply a static nonlinearity. TODO: use helper functions
    rather than a look-up table to determine which NL to apply. parameters can
    be saved in a generic vector self.phi - see NARF implementation for reference 
    
    @author: shofer
    """
    #Added helper functions and removed look up table --njs June 29 2017
    name='nonlinearity'
    plot_fns=[nu.pre_post_psth,nu.plot_spectrogram]
    user_editable_fields = ['nltype', 'fit_fields','phi']
    phi=np.array([1])
    
    def my_init(self,d_in=None,my_eval=None,nltype='dlog',fit_fields=['phi'],phi=[1],premodel=False):
        if premodel is True:
            self.do_plot=self.plot_fns[2]
        self.fit_fields=fit_fields
        self.nltype=nltype
        self.phi=np.array([phi])
        #setattr(self,nltype,phi0)
        if my_eval is None:
            if nltype=='dlog':
                self.my_eval=self.dlog_fn
                self.plot_fns=[nu.plot_spectrogram]
                self.do_plot=self.plot_fns[0]
            elif nltype=='exp':
                self.my_eval=self.exp_fn
            elif nltype=='dexp':
                self.my_eval=self.dexp_fn
        else:
            self.my_eval=my_eval
            
        
    def dlog_fn(self,X):
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

                 

