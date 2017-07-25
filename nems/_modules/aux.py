#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:48:25 2017

@author: shofer
"""
import lib.nems_utils as nu
import numpy as np
import scipy.stats as spstats
from nems_modules import nems_module
import copy


class normalize(nems_module):
    """
    normalize - rescale a variable, typically stim, to put it in a range that
    works well with fit algorithms --
    either mean 0, variance 1 (if sign doesn't matter) or
    min 0, max 1 (if positive values desired)
    IMPORTANT NOTE: normalization factors are computed from estimation data 
    only but applied to both estimation and validation data streams
    """
    name='normalize'
    user_editable_fields=['output_name','valfrac','valmode']
    force_positive=True
    d=0
    g=1
    
    def my_init(self, force_positive=True,data='stim'):
        self.force_positive=force_positive
        self.input_name=data
    
    def evaluate(self):
        X=self.unpack_data()
        name=self.input_name
        
        if self.d_in[0][name].ndim==2:
            if self.force_positive:
                self.d=X.min()
                self.g=1/(X-self.d).max()
            else:
                self.d=X.mean()
                self.g=X.std()
        else:
            s=self.d_in[0][name].shape
            if self.force_positive:
                self.d=X[:,:].min(axis=1).reshape([s[0],1,1])
                self.g=1/(X[:,:]-self.d.reshape([s[0],1])).max(axis=1).reshape([s[0],1,1])
            else:
                self.d=X[:,:].mean(axis=1).reshape([s[0],1,1])
                self.g=X[:,:].std(axis=1).reshape([s[0],1,1])
                self.g[np.isinf(g)]=0
                
        # apply the normalization
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
        
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            f_out[self.output_name]=np.multiply(X-self.d,self.g)

"""
modules for computing scores/ assessing model performance
"""
class mean_square_error(nems_module):
 
    name='mean_square_error'
    user_editable_fields=['input1','input2','norm']
    plot_fns=[nu.pred_act_psth,nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    norm=True
    shrink=0
    mse_est=np.ones([1,1])
    mse_val=np.ones([1,1])
        
    def my_init(self, input1='stim',input2='resp',norm=True,shrink=0):
        self.input1=input1
        self.input2=input2
        self.norm=norm
        self.shrink=shrink
        self.do_trial_plot=self.plot_fns[1]
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            
        if self.shrink:
            X1=self.unpack_data(self.input1,est=True)
            X2=self.unpack_data(self.input2,est=True)
            bounds=np.round(np.linspace(0,len(X1)+1,11)).astype(int)
            E=np.zeros([10,1])
            P=np.mean(np.square(X2))
            for ii in range(0,10):
                E[ii]=np.mean(np.square(X1[bounds[ii]:bounds[ii+1]]-X2[bounds[ii]:bounds[ii+1]]))
            E=E/P
            mE=E.mean()
            sE=E.std()
            if mE<1:
                # apply shrinkage filter to 1-E with factors self.shrink
                mse=1-nu.shrinkage(1-mE,sE,self.shrink)
            else:
                mse=mE
                
        else:
            E=np.zeros([1,1])
            P=np.zeros([1,1])
            N=0
            for f in self.d_out:
                #E+=np.sum(np.sum(np.sum(np.square(f[self.input1]-f[self.input2]))))
                E+=np.sum(np.square(f[self.input1]-f[self.input2]))
                #P+=np.sum(np.sum(np.sum(np.square(f[self.input2]))))
                P+=np.sum(np.square(f[self.input2]))
                N+=f[self.input2].size
        
            if self.norm:
                mse=E/P
            else:
                mse=E/N
                
        if self.parent_stack.valmode is True:   
            self.mse_val=mse
            self.parent_stack.meta['mse_val']=mse
        else:
            self.mse_est=mse
            self.parent_stack.meta['mse_est']=mse
        
        
        return mse

    def error(self, est=True):
        if est:
            return self.mse_est
        else:
            # placeholder for something that can distinguish between est and val
            return self.mse_val
        
class pseudo_huber_error(nems_module):
    """
    Pseudo-huber "error" to use with fitter cost functions. This is more robust to
    ouliers than simple mean square error. Approximates L1 error at large
    values of error, and MSE at low error values. Has the additional benefit (unlike L1)
    of being convex and differentiable at all places.
    
    Pseudo-huber equation taken from Hartley & Zimmerman, "Multiple View Geometry
    in Computer Vision," (Cambridge University Press, 2003), p619
    
    C(delta)=2(b^2)(sqrt(1+(delta/b)^2)-1)
    
    b mediates the value of error at which the the error is penalized linearly or quadratically.
    Note that setting b=1 is the soft l1 loss
    
    @author: shofer, June 30 2017
    """
    #I think this is working (but I'm not positive). When fitting with a pseudo-huber
    #cost function, the fitter tends to ignore areas of high spike rates, but does
    #a good job finding the mean spike rate at different times during a stimulus. 
    #This makes sense in that the huber error penalizes outliers, and could be 
    #potentially useful, depending on what is being fit? --njs, June 30 2017
    
    
    name='pseudo_huber_error'
    plot_fns=[nu.pred_act_psth,nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    b=0.9 #sets the value of error where fall-off goes from linear to quadratic\
    huber_est=np.ones([1,1])
    huber_val=np.ones([1,1])
    
    def my_init(self, input1='stim',input2='resp',b=0.9):
        self.input1=input1
        self.input2=input2
        self.b=b
        self.do_trial_plot=self.plot_fns[1]
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
    
        for f in self.d_out:
            delta=np.divide(np.sum(f[self.input1]-f[self.input2],axis=1),np.sum(f[self.input2],axis=1))
            C=np.sum(2*np.square(self.b)*(np.sqrt(1+np.square(np.divide(delta,self.b)))-1))
            C=np.array([C])
        self.huber_est=C
            
    def error(self,est=True):
        if est is True:
            return(self.huber_est)
        else: 
            return(self.huber_val)
        
            
        
class correlation(nems_module):
 
    name='correlation'
    user_editable_fields=['input1','input2']
    plot_fns=[nu.pred_act_psth, nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    r_est=np.ones([1,1])
    r_val=np.ones([1,1])
        
    def my_init(self, input1='stim',input2='resp',norm=True):
        self.input1=input1
        self.input2=input2
        self.do_plot=self.plot_fns[1]
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        X1=self.unpack_data(self.input1,est=True)            
        X2=self.unpack_data(self.input2,est=True)
        r_est,p=spstats.pearsonr(X1,X2)
        self.r_est=r_est
        self.parent_stack.meta['r_est']=r_est
                              
        X1=self.unpack_data(self.input1,est=False)            
        if X1.size:
            X2=self.unpack_data(self.input2,est=False)
            r_val,p=spstats.pearsonr(X1,X2)
            self.r_val=r_val
            self.parent_stack.meta['r_val']=r_val
        
            return r_val
        else:
            return (r_est)
    
 