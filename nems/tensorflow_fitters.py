#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:03:32 2017

@author: shofer
"""

import numpy as np
import scipy as sp
import tensorflow as tf


class tf_nems_fitter:
    """
    NEMS fitter class that utilizes Tensorflow backend. Accomplishes the same
    goal as nems_fitter class, but is structured quite differently internally.
    
    Note that for easy optimization, this will be slower that a scipy method, but
    should be faster for optimization with large arrays or many parameters.
    """
    # common properties for all modules
    name='default'
    phi0=None
    counter=0
    fit_modules=[]
    tol=0.001
    stack=None
    
    def __init__(self,parent,fit_modules=None,**xargs):
        self.stack=parent
        # figure out which modules have free parameters, if fit modules not specified
        if not fit_modules:
            self.fit_modules=[]
            for idx,m in enumerate(self.stack.modules):
                this_phi=m.parms2phi()
                if this_phi.size:
                    self.fit_modules.append(idx)
        else:
            self.fit_modules=fit_modules
        self.my_init(**xargs)
        
    def my_init(self,**xargs):
        pass
    
    def fit_to_phi(self):
        """
        Converts fit parameters to a single vector, to be used in fitting
        algorithms.
        """
        phi=[]
        for k in self.fit_modules:
            g=self.stack.modules[k].parms2phi()
            phi=np.append(phi,g)
        return(phi)
    
    def phi_to_fit(self,phi):
        """
        Converts single fit vector back to fit parameters so model can be calculated
        on fit update steps.
        """
        st=0
        for k in self.fit_modules:
            phi_old=self.stack.modules[k].parms2phi()
            s=phi_old.shape
            self.stack.modules[k].phi2parms(phi[st:(st+np.prod(s))])
            st+=np.prod(s)
            

    # create fitter, this should be turned into an object in the nems_fitters library
    def test_cost(self,phi):
        self.stack.modules[2].phi2parms(phi)
        self.stack.evaluate(1)
        self.counter+=1
        if self.counter % 100 == 0:
            print('Eval #{0}. MSE={1}'.format(self.counter,self.stack.error()))
        return self.stack.error()
    
    def do_fit(self):
        # run the fitter
        self.counter=0
        # pull out current phi as initial conditions
        self.phi0=self.stack.modules[1].parms2phi()
        phi=sp.optimize.fmin(self.test_cost, self.phi0, maxiter=1000)
        return phi
    
    
    
class ADADELTA_min(tf_nems_fitter):
    """
    Uses Tensorflow's ADADELTA algorithm to fit. This algorithm is particularly
    nice in that the learning rate is updated based on the gradient of the function
    computed at previous steps, along with an offset to prevent the learning rate
    form going to 0 prematurely.
    
    See: M.D. Zeiler, "ADADELTA: AN ADAPTIVE LEARNING RATE METHOD", arXiv:1212.5701v1, 2012
    """
    #tf.losses.mean_squared_error
    #tf.losses.huber_loss
    #tf.train.AdadeltaOptimizer
    
    def my_init(self,iters=3000):
        self.iters=iters
        print('Initializing ADADELTA fitter')
        
    def setup(phi):
        
        
        
    def do_fit(self):
        
        with tf.Session as sess:
            
            
    
    
    
    
    
    
    