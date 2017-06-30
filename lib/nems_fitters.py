#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import scipy as sp
import numpy as np
import sys
#sys.path.append('/auto/users/shofer/scikit-optimize')
#import skopt.optimizer.gbrt as skgb
#import skopt.optimizer.gp as skgp



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
        if self.routine=='L-BFGS-B':
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
    
#TODO: implement scipy basinhopping routine. Note that this is scipy's implementation 
#of simulated annealing


class anneal_min(nems_fitter):
    """
    A simulated annealing method to find the ~global~ minimum of your parameters.
    
    This fitter uses scipy.optimize.basinhopping, which is scipy's built-in annealing
    routine. Essentially, this routine uses scipy.optimize.minimize to minimize the
    function, then randomly perturbs the function and reminimizes it. It will continue
    this procedure until either the maximum number of iterations had been exceed or
    the minimum remains constant for a specified number of iterations.
    
    anneal_iter=number of annealing iterations to perform
    stop=number of iterations after which to stop annealing if global min remains the same
    up_int=update step size every up_int iterations
    maxiter=maximum iterations for each round of minimization
    tol=tolerance for each round of minimization
    min_method=method used for each round of minimization. 'L-BFGS-B' works well
    
    WARNING: this fitter takes a ~~long~~ time. It is usually better to try basic_min
    first, and then use this method if basic_min fails.
    
    Also, note that since basinhopping (at least as implemented here) uses random jumps,
    the results my not be exactly the same every time, and the annealing may take a 
    different number of iterations each time it is called
    """

    name='anneal_min'
    anneal_iter=100
    stop=5
    up_int=10
    min_method='L-BFGS-B'
    maxiter=10000
    tol=0.01
    
    def my_init(self,min_method='L-BFGS-B',anneal_iter=100,stop=5,maxiter=10000,up_int=10,bounds=None):
        print("initializing anneal_min")
        self.anneal_iter=anneal_iter
        self.min_method=min_method
        self.stop=stop
        self.maxiter=10000
        self.bounds=None
        self.up_int=up_int
                    
    def cost_fn(self,phi):
        self.phi_to_fit(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse=self.stack.error()
        self.counter+=1
        if self.counter % 1000==0:
            print('Eval #'+str(self.counter))
            print('MSE='+str(mse))
        return(mse)
    
    def do_fit(self,verb=False):
        opt=dict.fromkeys(['maxiter'])
        opt['maxiter']=int(self.maxiter)
        opt['eps']=1e-7
        min_kwargs=dict(method=self.min_method,tol=self.tol,bounds=self.bounds,options=opt)
        #if function=='tanhON':
            #cons=({'type':'ineq','fun':lambda x:np.array([x[0]-0.01,x[1]-0.01,-x[2]-1])})
            #routine='COBYLA'
        #else:
            #
        #cons=()
        self.phi0=self.fit_to_phi() 
        self.counter=0
        print("anneal_min: phi0 intialized (fitting {0} parameters)".format(len(self.phi0)))
        #print("maxiter: {0}".format(opt['maxiter']))
        opt_res=sp.optimize.basinhopping(self.cost_fn,self.phi0,niter=self.anneal_iter,
                                         T=0.01,stepsize=0.01,minimizer_kwargs=min_kwargs,
                                         interval=self.up_int,disp=verb,niter_success=self.stop)
        phi_final=opt_res.lowest_optimization_result.x
        self.cost_fn(phi_final)
        print("Final MSE: {0}".format(self.stack.error()))
        return(self.stack.error())

"""
Tried using skopt package. Did not go super well, only used pupil data though.
Will try again later with different data (i.e. more estimation data) --njs, June 29 2017
    
class forest_min(nems_fitter):
    name='forest_min'
    maxit=100
    routine='skopt_ft'
    
    def my_init(self,dims,maxit=500):
        print("initializing basic_min")
        self.maxit=maxit
        self.dims=dims
        
                    
    def cost_fn(self,phi):
        #print(phi.shape)
        phi=np.array(phi)
        self.phi_to_fit(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse=self.stack.error()
        self.counter+=1
        mse=np.asscalar(mse)
        if self.counter % 100==0:
            print('Eval #'+str(self.counter))
            print('MSE='+str(mse))
        #print(mse)
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
        #cons=()
        self.phi0=np.array(self.fit_to_phi())
        self.y0=self.cost_fn(self.phi0)
        self.counter=0
        print("gaussian_min: phi0 intialized (fitting {0} parameters)".format(len(self.phi0)))
        #print("maxiter: {0}".format(opt['maxiter']))
        #sp.optimize.minimize(self.cost_fn,self.phi0,method=self.routine,
                             #constraints=cons,options=opt,tol=self.tol)
        #skgp.gp_minimize(self.cost_fn,self.dims,base_estimator=None, n_calls=100, 
                         #n_random_starts=10, acq_func='gp_hedge', acq_optimizer='auto', x0=self.phi0, 
                         #y0=self.y0, random_state=True, verbose=True)
        skgb.gbrt_minimize(func=self.cost_fn,dimensions=self.dims,n_calls=self.maxit,x0=self.phi0,
                         y0=self.y0,random_state=False,verbose=True)
        print("Final MSE: {0}".format(self.stack.error()))
        return(self.stack.error())
"""

class coordinate_descent(nems_fitter):
    """
    coordinate descent - step one parameter at a time
    """

    name='coordinate_descent'
    maxit=1000
    tol=0.001
    step_init=0.01
    step_change=np.sqrt(0.5)
    step_min=1e-7
    
    
    def my_init(self,tol=0.001,maxit=1000):
        print("initializing basic_min")
        self.maxit=maxit
        self.tol=tol
                    
    def cost_fn(self,phi):
        self.phi_to_fit(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse=self.stack.error()
        self.counter+=1
        if self.counter % 100==0:
            print('Eval #{0}: Error={1}'.format(self.counter,mse))
        return(mse)
    
    def do_fit(self):
        
        self.phi0=self.fit_to_phi() 
        self.counter=0
        n_params=len(self.phi0)
        
#        if ~options.Elitism
#            options.EliteParams = n_params;
#            options.EliteSteps = 1;
#        end

        n = 1;   # step counter
        x = self.phi0.copy(); # current phi
        x_save = x.copy()     # last updated phi
        s = self.cost_fn(x)   # current score
        s_new = np.zeros([n_params,2])     # Improvement of score over the previous step
        s_delta = np.inf     # Improvement of score over the previous step
        step_size = self.step_init;  # Starting step size.
        #print("{0}: phi0 intialized (start error={1}, {2} parameters)".format(self.name,s,len(self.phi0)))
        #print(x)
      
        while s_delta>self.tol and n<self.maxit and step_size>self.step_min:
            for ii in range(0,n_params):
                for ss in [0,1]:
                    x[:]=x_save[:]
                    if ss==0:
                        x[ii]+=step_size
                    else:
                        x[ii]-=step_size
                    s_new[ii,ss]=self.cost_fn(x)
                    
            x_opt=np.unravel_index(s_new.argmin(),(n_params,2))
            popt,sopt=x_opt
            s_delta=s-s_new[x_opt]
            print("{0}: best step={1},{2} error={3}, delta={4}".format(n,popt,sopt,s_new[x_opt],s_delta))
            
            if s_delta<0:
                step_size=step_size*self.step_change
                print("Backwards, adjusting step size to {0}".format(step_size))
            
            elif s_delta<self.tol:
                print("Error improvement too small. Iteration complete.")
                
            elif sopt:
                x_save[popt]-=step_size
            else:
                x_save[popt]+=step_size
            
            x=x_save.copy()
            n+=1
            s=s_new[x_opt]
            
        # save final parameters back to model
        self.phi_to_fit(x)
        
        #print("Final MSE: {0}".format(s))
        return(s)



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
        self.sub_fitter.tol=self.sub_fitter.tol
        iter=0
        err=self.stack.error()
        while iter<self.max_iter:
            for i in self.fit_modules:
                print("Begin sub_fitter on mod {0}/iter {1}/tol={2}".format(i,iter,self.sub_fitter.tol))
                self.sub_fitter.fit_modules=[i]
                new_err=self.sub_fitter.do_fit()
            if err-new_err<self.sub_fitter.tol:
                print("")
                print("error improvement less than tol, starting new outer iteration")
                iter+=1
                self.sub_fitter.tol=self.sub_fitter.tol/2
            err=new_err
            
        return(self.stack.error())


    