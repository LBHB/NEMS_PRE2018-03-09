#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import scipy as sp
import numpy as np

class nems_fitter:
    """nems_fitter
    
    Generic NEMS fitter object
    
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
        to_fit should be formatted ['par1','par2',] etc
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
        to_fit should be formatted ['par1','par2',] etc
        """
        st=0
        for k in self.fit_modules:
            phi_old=self.stack.modules[k].parms2phi()
            s=phi_old.shape
            self.stack.modules[k].phi2parms(phi[st:(st+np.prod(s))])
            st+=np.prod(s)
            

    # create fitter, this should be turned into an object in the nems_fitters libarry
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
    
    
class basic_min(nems_fitter):
    """
    The basic fitting routine used to fit a model. This function defines a cost
    function that evals the functions being fit using the current parameters
    for those functions and outputs the current mean square error (mse). 
    This cost function is evaulated by the scipy optimize.minimize routine,
    which seeks to minimize the mse by changing the function parameters. 
    This function has err, fit_to_phi, and phi_to_fit as dependencies.
    
    Scipy optimize.minimize is set to use the minimization algorithm 'L-BFGS-B'
    as a default, as this is memory efficient for large parameter counts and seems
    to produce good results. However, there are many other potential algorithms 
    detailed in the documentation for optimize.minimize
    
    Function returns self.pred, the current model prediction. 
    """

    name='basic_min'
    maxit=50000
    routine='L-BFGS-B'
    
    def my_init(self,routine='L-BFGS-B',maxit=50000):
        print("initializing basic_min")
        self.maxit=maxit
        self.routine=routine
                    
    def cost_fn(self,phi):
        self.phi_to_fit(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse=self.stack.error()
        self.counter+=1
        if self.counter % 1000==0:
            print('Eval #'+str(self.counter))
            print('MSE='+str(mse))
        return(mse)
    
    def do_fit(self):
        
        opt=dict.fromkeys(['maxiter'])
        opt['maxiter']=int(self.maxit)
        opt['eps']=1e-7
        #if function=='tanhON':
            #cons=({'type':'ineq','fun':lambda x:np.array([x[0]-0.01,x[1]-0.01,-x[2]-1])})
            #routine='COBYLA'
        #else:
            #
        cons=()
        self.phi0=self.fit_to_phi() 
        self.counter=0
        print("basic_min: phi0 intialized (fitting {0} parameters)".format(len(self.phi0)))
        #print("maxiter: {0}".format(opt['maxiter']))
        sp.optimize.minimize(self.cost_fn,self.phi0,method=self.routine,
                             constraints=cons,options=opt,tol=self.tol)
        print("Final MSE: {0}".format(self.stack.error()))
        return(self.stack.error())

class fit_iteratively(nems_fitter):
    """
    iterate through modules, running fitting each one with sub_fitter()
    """

    name='fit_iteratively'
    sub_fitter=None
    max_iter=5
    
    def my_init(self,sub_fitter=basic_min,max_iter=5):
        self.sub_fitter=sub_fitter(self.stack)
        self.max_iter=max_iter
            
    def do_fit(self):
        self.sub_fitter.tol=self.sub_fitter.tol*2
        for iter in range(0,self.max_iter):
            self.sub_fitter.tol=self.sub_fitter.tol/2
            for i in self.fit_modules:
                print("Begin sub_fitter on mod {0}/iter {1}/tol={2}".format(i,iter,self.sub_fitter.tol))
                self.sub_fitter.fit_modules=[i]
                self.sub_fitter.do_fit()
        
        return(self.stack.error())


    